# Random Sample Bang Backend Core Implementation Documentation

该模块实现了 Infini 框架中随机采样算子的寒武纪 MLU（Bang 架构）后端，支持 argmax、top-k、top-p (nucleus) 和温度采样等多种采样策略，为大语言模型推理提供高效的 token 生成能力。

## 1. Module Structure

- **`random_sample_bang.h`**: 算子描述符声明与宏定义接口，通过 DESCRIPTOR 宏生成 Descriptor 类
- **`random_sample_bang.mlu`**: 算子描述符实现，包含 workspace 计算与 kernel 调度逻辑
- **`random_sample_kernel.mlu`**: MLU 设备代码实现，包含所有设备端的采样 kernel 函数与算法

## 2. Core Classes

### `Descriptor`
- **Location**: `random_sample_bang.h` (宏定义), `random_sample_bang.mlu` (实例化)
- **Primary Function**: 封装随机采样算子的 Bang 后端实现，管理算子生命周期、workspace 内存分配和 kernel 执行调度
- **Key Members**:
  - `_opaque`: `Opaque*` 类型，持有 `device::bang::Handle::Internal` 共享指针，用于管理 MLU 设备句柄
  - `_info`: `RandomSampleInfo` 结构，存储张量数据类型信息（索引类型 `dt_i`、概率类型 `dt_p`）、词汇表大小 `n`
  - `_min_workspace_size`: `size_t` 类型，kernel 执行所需的最小 workspace 字节数
- **Core Methods**:
  - `create(handle_, desc_ptr, result_desc, probs_desc)`: 静态工厂方法，验证输入张量描述符、根据数据类型组合实例化模板化的 `calculateWorkspace` 函数计算 workspace 大小，构造 Descriptor 对象并返回成功状态
  - `minWorkspaceSize()`: 返回 `_min_workspace_size`，供上层框架预分配设备内存
  - `calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream)`: 主执行函数，验证 workspace 大写是否充足，调用 `Calculate::calculate` 分发到 `Algo` 结构体的 `argmax` 或 `random` 方法
- **Lifecycle**: 通过 `create` 静态方法构造，持有 Bang 设备句柄的共享指针以延长设备生命周期，析构时释放 `_opaque` 指针

### `Algo`
- **Location**: `random_sample_kernel.mlu`
- **Primary Function**: 策略类，提供 argmax 和随机采样两种算法的模板化实现，通过编译期类型派发到不同的 MLU kernel 函数
- **Core Methods**:
  - `argmax<Tidx, Tval_>(workspace, workspace_size, result_, probs, voc, stream_)`: 当 `random_val==0` 或 `topp==0` 或 `topk==1` 或 `temperature==0` 时退化到贪心解码，启动配置为 `{4, 1, 1}` 的 4-cluster 任务块，调用 `argMax` kernel 执行并行归约求最大值索引，同步队列后返回
  - `random<Tidx, Tval_>(workspace, workspace_size, result_, probs, voc, random_val, topp, topk, temperature, stream_)`: 执行随机采样，根据词汇表大小 `voc` 与单次处理能力 `SRC_MAX_SIZE/sizeof(Tval)` 的比较，选择 `randomSampleKernelLarge`（分多轮处理）或 `randomSampleKernel`（单轮处理），启动配置为 `{4, 1, 1}`，使用 `CNRT_FUNC_TYPE_UNION1` 联合计算类型，同步队列后返回
- **Implementation Details**: 使用 `if constexpr` 编译期类型判断支持 `float`、`CustomFloat16` (half)、`CustomBFloat16` (bfloat16_t) 三种值类型，每种类型需重新计算 workspace 偏移（`gdram_indices` 大小为 `sizeof(Tidx)*voc`，`global_top_k` 大小为 `task_num*topk`，`global_sum` 大小为 1）

### Device Kernel Functions
所有 kernel 函数均位于 `random_sample_kernel.mlu`，使用 `__mlu_global__` 声明：

#### `argMax<Tval, Tidx>(probs, result, gdram_indices, vocab_size)`
- **Algorithm**: 分块并行归约求 argmax
- **Implementation**:
  1. 将词汇表按 `taskDim * max_num`（max_num = 32768/sizeof(Tval)）分块，每个 task 处理 `max_num` 个元素
  2. 对完整块：通过 `__memcpy` 从 GDRAM 加载到 NRAM，调用 Bang API `__bang_argmax` 找出局部最大值和索引，维护当前全局最大值 `current_max` 和索引 `current_index`
  3. 对余数部分：根据 taskId 不均匀分配（部分 task 多处理一个元素），填充 -inf 后同样调用 `__bang_argmax`
  4. 各 task 将结果写入 `gdram_indices[taskId]`
  5. task 0 遍历所有 task 的结果，找出全局最大值（对 bfloat16 使用 `to_float` 比较），写入 `result[0]`
- **Complexity**: O(vocab_size / taskDim) 时间，O(max_num) NRAM 空间

#### `randomSampleKernel<Tval, Tidx>(probs, result, gdram_indices, global_topk, global_sum, vocab_size, random_val, topp, topk, temperature)`
- **Algorithm**: 单轮 top-k + softmax + top-p 采样（适用于 vocab_size < taskDim * max_num）
- **Implementation**:
  1. **温度缩放逆计算**: 对 bfloat16 使用 `to_bfloat16(1.0f/temperature)`，其他类型直接类型转换
  2. **负载均衡**: `step = vocab_size/taskDim + (taskId < vocab_size%taskDim ? 1 : 0)`，`start_idx = taskId*(vocab_size/taskDim) + min(taskId, vocab_size%taskDim)`
  3. **Top-k 聚合**:
     - 从 GDRAM 加载本 task 负责的概率片段到 `nram_src`
     - 初始化索引数组 `nram_indices` 为 `[start_idx, start_idx+1, ...]`
     - 若实际元素 `step < topk`，调用 `initTopkBuffer` 用 -inf 和 -1 填充剩余位置
     - 调用 `findTopk` 执行选择排序（O(k^2)）找出本 task 的 top-k 元素
     - 各 task 将结果写入 `global_topk[taskId*topk : (taskId+1)*topk]` 和 `gdram_indices`
     - `__sync_all()` 同步后，task 0 将所有 task 的 top-k 合并，再次调用 `findTopk` 得到全局 top-k
  4. **Softmax 计算**:
     - task 0 重置 `global_sum[0] = 0`
     - 所有 task 重新加载原始概率，执行稳定 softmax：`probs - global_max`，`* temp_inv`，手动 clamp 到 20.0（防止 exp 溢出），`__bang_active_exp_less_0` 计算 exp，累加到 `nram_partial_sum`
     - 二分归约求和：通过 strip 循环将 `max_num` 个元素折叠到 `w_size` (128/sizeof(Tval)) 个，调用 `__bang_reduce_sum` 得到标量和
     - `__bang_atomic_add` 原子累加到 `global_sum[0]`
  5. **Top-p 采样** (仅 task 0):
     - 检查 `global_sum[0] <= 0`，若是则回退到选择第一个索引
     - 计算逆和 `global_sum_inv = 1.0/global_sum[0]`
     - 重新为 top-k 元素计算 softmax（温度缩放、exp、归一化）
     - 计算累积和，找到满足 `cumsum >= topp` 的最小 `end`
     - 将 `random_val` 缩放到 `[0, cumsum]` 范围
     - 遍历累积和，找到第一个满足 `random_val < cumsum` 的索引，写入 `result[0]`
     - 若未命中（数值误差），回退到 `gdram_indices[end-1]`
- **NRAM Layout**:
  - `nram_src`: max_num 个 Tval（输入/工作缓冲）
  - `nram_partial_sum`: max_num 个 Tval（分块和）
  - `nram_sum_final`: w_size 个 Tval（最终和）
  - `nram_topk`: topk 个 Tval（top-k 值）
  - `nram_indices`: max_num 个 Tidx（索引缓冲）
  - `nram_global_indices`: max_num 个 Tidx（全局索引）

#### `randomSampleKernelLarge<Tval, Tidx>(probs, result, gdram_indices, global_topk, global_sum, vocab_size, random_val, topp, topk, temperature)`
- **Algorithm**: 多轮分块 top-k + softmax + top-p 采样（适用于 vocab_size >= taskDim * max_num）
- **Implementation**:
  1. **负载均衡**: 将词汇表分为 `repeat = vocab_size/task_size` 个完整块（task_size = taskDim * max_num）和一个余数部分
     - 余数部分不均匀分配：`remain_t = remain % taskDim`，`step_hard = step_easy + 1`，前 `remain_t` 个 task 处理 `step_hard` 个元素，其余处理 `step_easy`
     - `start_idx = (taskId < remain_t) ? taskId*step_hard : remain_t*step_hard + (taskId-remain_t)*step_easy`
  2. **增量式 Top-k 聚合**:
     - 初始化 `nram_topk_buffer[2*topk]` 为 -inf，`nram_indices` 为 `[taskId*max_num, ...]`
     - 对每个完整块：
       - 更新索引：`__bang_add_scalar((short*)nram_indices, ..., task_size, max_num/sizeof(short))`（步进一个块）
       - 加载概率到 `nram_src`，调用 `findTopk` 找出本块 top-k
       - 将结果追加到 `nram_topk_buffer[topk:2*topk]` 和 `nram_topk_indices`，调用 `findTopk` 合并前 2*topk 个元素，保留 top-k
     - 对余数部分：
       - 重置 `nram_indices` 为实际起始索引，加载剩余概率
       - 若 `step >= topk`，直接调用 `findTopk`；否则调用 `initTopkBuffer` 填充 -inf
       - 追加到缓冲区并合并
     - `__sync_all()` 后 task 0 合并所有 task 的 top-k 结果
  3. **Softmax 计算**:
     - task 0 重置 `global_sum[0] = 0`
     - 所有 task 遍历完整块和余数部分，加载概率、温度缩放、exp、累加
     - 若 `max_num >= w_size`，使用 strip 循环二分归约；否则直接线性累加（对 bfloat16 使用 `to_float` 加法）
     - `__sync_all()` + `__bang_atomic_add` 累加到 `global_sum[0]`
  4. **Top-p 采样** (仅 task 0):
     - 重新计算 top-k 的 softmax（温度缩放、exp、归一化）
     - 计算累积和：`nram_topk_buffer[i] = nram_topk_buffer[i-1] + nram_global_topk[i]`（对 bfloat16 使用 `to_float` 加法）
     - 找到满足 `cumsum >= topp` 的 `end`，处理边界条件 `end < topk-1 ? end+1 : topk`
     - 缩放 `random_val *= cumsum[end-1]`
     - 遍历累积和，找到第一个满足 `random_val < cumsum` 的索引
- **NRAM Layout**:
  - 通过 `nram_buffer_ind` 偏移分离索引缓冲区和值缓冲区
  - `nram_topk_buffer`: 2*topk 个 Tval（增量 top-k 合并缓冲）
  - `nram_global_topk`: taskDim*topk 个 Tval（全局 top-k）
  - `nram_topk_indices`: 2*topk 个 Tidx（增量索引合并）
  - `nram_global_indices`: taskDim*topk 个 Tidx（全局索引）

#### Helper Functions
- **`swap<Tval>(a, b)`**: 设备端交换函数，使用临时变量
- **`findTopk<Tval, Tidx>(values, result, size, topk)`**: 选择排序实现 top-k，时间复杂度 O(topk * size)，空间复杂度 O(1)，通过比较 `values[i] < values[j]` 同时交换值和索引
- **`initTopkBuffer<Tval, Tidx>(values, result, actual_size, topk)`**: 当 `actual_size < topk` 时，使用 `__bang_write_value` 填充 `values[actual_size:topk]` 为 -inf，`result[actual_size:topk]` 为 -1，保证后续算法稳定性
- **`calculateWorkspace<Tidx, Tval>(n)`**: 返回 `n * (sizeof(Tidx) + sizeof(Tval)) + sizeof(Tval)` 字节，覆盖 `gdram_indices`、`global_topk`、`global_sum` 的总和

## 3. API Interface

```cpp
// 创建算子描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                  // Bang 设备句柄
    Descriptor **desc_ptr,                     // 输出：描述符指针
    infiniopTensorDescriptor_t result_desc,    // 输出张量描述符（标量，整数类型）
    infiniopTensorDescriptor_t probs_desc);    // 输入概率张量描述符（1D，浮点类型）

// 查询最小 workspace 大小
size_t Descriptor::minWorkspaceSize() const;

// 执行采样计算
infiniStatus_t Descriptor::calculate(
    void *workspace,           // 设备 workspace 指针
    size_t workspace_size,     // workspace 字节大小
    void *result,              // 输出：设备内存，单个索引值
    const void *probs,         // 输入：设备内存，概率分布
    float random_val,          // 随机数 [0, 1)，若为 0 则执行 argmax
    float topp,                // top-p 阈值 [0, 1]，若为 0 则执行 argmax
    int topk,                  // top-k 大小，若为 1 则执行 argmax
    float temperature,         // 温度参数，若为 0 则执行 argmax
    void *stream) const;       // Bang 计算流
```

## 4. Usage Example

```cpp
// 初始化 Bang 设备句柄（假设已创建）
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_BANG, 0);

// 准备张量描述符
int64_t dims[] = {vocab_size};
infiniopTensorDescriptor_t probs_desc, result_desc;
infiniopCreateTensorDescriptor(&probs_desc, INFINI_DTYPE_F32, 1, dims, nullptr);
infiniopCreateTensorDescriptor(&result_desc, INFINI_DTYPE_I32, 0, nullptr, nullptr);

// 创建算子描述符
op::random_sample::bang::Descriptor *desc;
auto status = op::random_sample::bang::Descriptor::create(
    handle, &desc, result_desc, probs_desc);

// 分配 workspace 和设备内存
size_t workspace_size = desc->minWorkspaceSize();
void *workspace, *d_probs, *d_result;
cnrtMalloc(&workspace, workspace_size);
cnrtMalloc(&d_probs, vocab_size * sizeof(float));
cnrtMalloc(&d_result, sizeof(int32_t));

// 上传概率数据（假设 h_probs 为主机端数据）
cnrtMemcpy(d_probs, h_probs, vocab_size * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);

// 创建计算流
cnrtQueue_t stream;
cnrtQueueCreate(&stream);

// 执行采样（例如：top-50、top-p=0.9、温度=1.0）
float random_val = 0.374f;  // 从均匀分布生成
status = desc->calculate(
    workspace, workspace_size,
    d_result, d_probs,
    random_val, 0.9f, 50, 1.0f, stream);

// 同步并取回结果
cnrtQueueSync(stream);
int32_t token;
cnrtMemcpy(&token, d_result, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);

printf("Sampled token: %d\n", token);

// 清理资源
cnrtFree(d_result);
cnrtFree(d_probs);
cnrtFree(workspace);
cnrtQueueDestroy(stream);
delete desc;
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### Memory Management
- **Workspace 策略**: 单一连续设备内存，按偏移划分为三个区域：`gdram_indices`（大小 `sizeof(Tidx)*vocab_size`，存储中间索引）、`global_topk`（大小 `taskDim*topk*sizeof(Tval)`，存储各 task 的 top-k 值）、`global_sum`（大小 `sizeof(Tval)`，存储全局 softmax 和）
- **NRAM 管理**: 使用全局 `__nram__ char nram_buffer[NRAM_MAX_SIZE]`（256KB），通过指针算术分配多个子缓冲区（源数据、部分和、最终和、top-k、索引），利用 Bang 架构的片上内存实现高带宽访问
- **数据类型适配**: 对 bfloat16 使用 `to_float`/`to_bfloat16` 转换函数保证数值稳定性，避免直接运算导致的精度损失

### Concurrency
- **并行模型**: Cluster 级并行（启动配置 `{4, 1, 1}` 表示 4 个 cluster），每个 cluster 内部通过 `taskDim`（等于 cluster 数量）和 `taskId` 进行任务分块
- **同步原语**:
  - `__sync_all()`: 任务栅栏，确保所有 task 完成当前阶段（如写入全局 top-k 缓冲区）
  - `__sync_io()`: IO 同步，确保全局内存写入对其他 task 可见
  - `__sync_compute()`: 计算同步，确保局部计算完成
  - `__bang_atomic_add(ptr, val, count)`: 原子加法，用于多个 task 累加到 `global_sum[0]`
- **负载均衡**: 采用非均匀分块策略（前 `remain%taskDim` 个 task 多处理一个元素），保证余数部分均匀分布，避免负载倾斜

### Performance
- **算法选择**: 根据 vocab_size 动态选择 kernel：
  - 小词汇表（< 4*32768/sizeof(Tval)）：使用 `randomSampleKernel`，单轮加载全部数据
  - 大词汇表（≥ 4*32768/sizeof(Tval)）：使用 `randomSampleKernelLarge`，多轮分块加载，增量式 top-k 合并
- **归约优化**: Softmax 求和采用 strip 循环二分折叠（类似并行归约树），将 `max_num` 个元素逐步减少到 `w_size`（128/sizeof(Tval)），最终调用 `__bang_reduce_sum` 得到标量，充分利用向量计算单元
- **Top-k 算法**: 选择排序实现 O(k^2)，在 k 较小（通常 50-100）时性能可接受，且对 NRAM 友好（原地排序，无需额外内存）
- **数值稳定性**: Softmax 计算采用 max 减法防止溢出，手动 clamp 到 20.0 防止 exp 上溢，对 bfloat16 使用 float 中间运算避免下溢

### Error Handling
- **Workspace 检查**: `calculate` 函数首先验证 `workspace_size >= _min_workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **数据类型验证**: `create` 函数通过 `RandomSampleInfo::create` 检查 `dt_i` 必须为整数类型，`dt_p` 必须为浮点类型（F16/BF16/F32/F64），`result_desc` 必须为 0 维标量，`probs_desc` 必须为 1 维且连续（stride(0)==1）
- **回退策略**:
  - 若 `global_sum[0] <= 0`（softmax 和异常），直接选择第一个 top-k 索引
  - 若 top-p 采样未命中（数值误差），回退到累积和的最后一个索引
  - 若温度或 top-p/topk 为 0，退化到 argmax 贪心解码

### Dependencies
- **Bang Runtime API (BANG)**:
  - `__memcpy`: 数据传输（GDRAM↔NRAM，支持 GDRAM2NRAM、NRAM2GDRAM、NRAM2NRAM）
  - `__bang_write_value`: 向量赋值（填充标量）
  - `__bang_argmax`: 向量 argmax（返回 `{max_value, max_index}` 对）
  - `__bang_sub_scalar`, `__bang_mul_scalar`: 向量-标量运算
  - `__bang_active_exp_less_0`: 条件激活函数（x < 0 ? exp(x) : x）
  - `__bang_add`, `__bang_reduce_sum`: 向量加法和归约求和
  - `__bang_atomic_add`: 原子加法
  - `cnrtDim3_t`, `CNRT_FUNC_TYPE_BLOCK`, `CNRT_FUNC_TYPE_UNION1`, `<<<>>>`: Kernel 启动语法
  - `cnrtQueue_t`, `cnrtQueueSync`: 计算流管理
- **Custom Types**: `CustomFloat16`, `CustomBFloat16`, `to_float`, `to_bfloat16`（来自 `custom_types.h`）
- **Common Headers**: `bang_kernel_common.h`（通用 kernel 工具），`common_bang.h`（Bang 通用宏），`infinicore.h`（Infini 核心头文件）

### Design Patterns
- **Strategy Pattern**: `Algo` 结构体作为策略类，通过模板方法 `argmax` 和 `random` 封装不同采样算法，`Calculate::calculate` 根据参数选择策略
- **Template Method Pattern**: `DESCRIPTOR` 宏生成模板化 `Descriptor` 类，定义构造函数、`create`、`minWorkspaceSize`、`calculate` 等骨架方法，由具体后端（如 `bang`）实例化
- **CRTP (Curiously Recurring Template Pattern)**: 隐式体现在 `Calculate::calculate` 接受 `Algo` 参数并调用其模板方法，编译期静态派发
- **Opaque Pointer Pattern**: `Descriptor` 通过 `Opaque` 结构体隐藏设备句柄实现细节，保持接口稳定
- **RAII**: 使用 `std::shared_ptr<device::bang::Handle::Internal>` 自动管理设备句柄生命周期

### Hardware-Specific Optimizations
- **NRAM 容量**: `SRC_MAX_SIZE = 32KB`，适配寒武纪 MLU 的片上内存限制，支持每轮加载 32768 个 float 或 65536 个 half/bfloat16
- **Cluster 启动**: 固定使用 4-cluster 配置（`dim = {4, 1, 1}`），利用 MLU 多 cluster 并行能力
- **Union 计算类型**: 使用 `CNRT_FUNC_TYPE_UNION1` 启用联合计算，提高 cluster 内并行度
- **非均匀分块**: 处理余数部分时，前 `remain%taskDim` 个 task 多处理一个元素，避免最后一个 task 负载过重
- **BFloat16 特殊处理**: 所有比较和加法操作使用 `to_float` 转换，避免 bfloat16 精度损失导致的累积误差
