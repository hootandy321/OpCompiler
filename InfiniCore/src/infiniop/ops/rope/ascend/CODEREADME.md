# RoPE (Rotary Position Embedding) Ascend AI Core Kernel Implementation

华为昇腾(Ascend) NPU的RoPE算子实现，基于AscendC编程模型，利用AI Core的向量计算单元对Transformer位置编码进行高效旋转操作。本实现针对GPT-J算法优化，支持FP16和FP32数据类型，通过双缓冲流水线机制最大化硬件吞吐量。

## 1. Module Structure

- **`rope_ascend.h`**: 外部接口声明，定义 `rope_kernel_launch` 函数签名和Descriptor宏，作为主机端与AI Core Kernel的桥梁
- **`rope_ascend.cc`**: Descriptor实现类，负责参数验证、RoPEInfo元数据构建、kernel启动调度，是主机端控制逻辑的核心
- **`rope_ascend_kernel.cpp`**: AscendC AI Core Kernel实现，包含 `RoPEKernel` 模板类和双缓冲流水线操作，实际执行旋转位置编码的向量计算

## 2. Core Classes

### `RoPEKernel<T, U>`
- **Location**: `rope_ascend_kernel.cpp:6-62`
- **Primary Function**: AscendC AI Core向量计算Kernel，实现RoPE算法的并行计算，每个Block处理一个注意力头(head)
- **Template Parameters**:
  - `T`: 数据类型 (float/half)，用于输入x、输出y、sin/cos表
  - `U`: 位置ID类型 (int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t)
- **Key Members**:
  - `TPipe pipe`: 流水线管理器，协调输入/输出队列的数据搬运
  - `TQue<QuePosition::VECIN, BUFFER_NUM> _in_que, _sin_que, _cos_que`: 输入双缓冲队列，VECIN表示向量输入缓冲区，BUFFER_NUM=2实现ping-pong缓冲
  - `TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que`: 输出双缓冲队列
  - `TBuf<TPosition::VECCALC> _tmp_odd_buf, _tmp_even_buf, _tmp_odd_buf1/2, _tmp_even_buf1/2`: 向量计算缓冲区，分别存储奇/偶索引元素及其中间结果
  - `GlobalTensor<T> _x_gm, _y_gm, _sin_gm, _cos_gm`: 全局内存(Global Memory)指针，指向HBM/HBM中的数据
  - `GlobalTensor<U> _p_gm`: 位置ID全局内存指针
  - `size_t _block_idx, _tile_len, _copy_len, _half_copy_len`: 块索引、分片长度(=dhead)、对齐后的拷贝长度
  - `ptrdiff_t _st_ynt, _st_ynh, _st_xnt, _st_xnh`: 输出/输入的stride(seqlen/nhead维度)
- **Core Methods**:
  - `init(GM_ADDR y, x, pos, sin, cos, size_t dh, ptrdiff_t st_ynt, st_ynh, st_xnt, st_xnh)`: 初始化全局内存地址、队列缓冲区、临时计算buffer，计算对齐后的拷贝长度(32字节对齐)，调用 `alignTileLen<T>(dh, BYTE_ALIGN)` 确保满足AI Core的DMA搬运要求
  - `process(size_t seq_len)`: 主循环，遍历每个序列位置 `i`，依次调用 `copyIn(i) -> compute(i) -> copyOut(i)` 实现流水线并行
  - `copyIn(size_t i)`: 从Global Memory搬运数据到Unified Buffer (UB)，包括：
    1. 输入向量x: `_x_gm[idx]` (idx = i*st_xnt + block_idx*st_xnh)
    2. sin/cos表: 根据位置ID `pos_idx = _p_gm(i)` 索引 `sin_gm[pos_idx * tile_len/2]` 和 `cos_gm[pos_idx * tile_len/2]`
    3. 数据搬运使用 `DataCopy` API，长度为 `_copy_len` (可能包含padding)
  - `compute(size_t i)`: 执行RoPE核心计算，包含以下步骤：
    1. **奇偶分离**: 使用 `GatherMask` 分别收集奇/偶索引元素 (stride=2)
       - `GatherMask<T>(tmp_odd, input_ub, 1, false, 0, gMaskParams, rsvdCnt)`: 收集索引0,2,4...
       - `GatherMask<T>(tmp_even, input_ub, 2, false, 0, gMaskParams, rsvdCnt)`: 收集索引1,3,5...
       - `GatherMaskParams`: mask_count=1, repeat_count=ceil((tile_len*sizeof(T))/256), repeat_stride=8
    2. **旋转计算** (GPT-J算法):
       - 奇索引元素: `y_odd = x_odd * cos - x_even * sin`
         - `Mul<T>(tmp_odd1, tmp_odd, cos_ub, tile_len/2)`
         - `Mul<T>(tmp_odd2, tmp_even, sin_ub, tile_len/2)`
         - `Sub<T>(tmp_odd1, tmp_odd1, tmp_odd2, tile_len/2)`
       - 偶索引元素: `y_even = x_odd * sin + x_even * cos`
         - `Mul<T>(tmp_even1, tmp_odd, sin_ub, tile_len/2)`
         - `Mul<T>(tmp_even2, tmp_even, cos_ub, tile_len/2)`
         - `Add<T>(tmp_even1, tmp_even1, tmp_even2, tile_len/2)`
    3. **奇偶合并**: 逐元素写入output_ub: `output_ub(j*2)=tmp_odd1(j), output_ub(j*2+1)=tmp_even1(j)`
    4. `PipeBarrier<PIPE_V>()`: 确保向量计算完成后再进行后续操作
  - `copyOut(size_t i)`: 将结果从UB搬运回Global Memory，使用 `DataCopyPad` 处理可能的padding，目标地址为 `_y_gm[idy]` (idy = i*st_ynt + block_idx*st_ynh)
- **Lifecycle**:
  - 每个 `RoPEKernel` 实例对应一个AI Core Block，由 `<<<nhead, nullptr, stream>>>` 启动时自动创建
  - 通过 `GetBlockIdx()` 获取当前Block索引，用于确定处理哪个head
  - 生命周期仅限于单次kernel launch，无持久化状态

### `op::rope::ascend::Descriptor`
- **Location**: `rope_ascend.cc:4-55`
- **Primary Function**: 主机端(HOST)描述符类，继承自 `InfiniopDescriptor`，负责参数校验、RoPEInfo元数据提取、workspace大小计算、kernel启动调度
- **Inheritance**: 继承 `InfiniopDescriptor` (定义于 `rope.h:12-54`，通过 `DESCRIPTOR(ascend)` 宏展开生成)
- **Key Members**:
  - `RoPEInfo _info`: 张量形状、stride、数据类型等元数据
  - `size_t _workspace_size`: workspace大小(当前实现为0)
  - `Opaque *_opaque`: 预留给future扩展的私有数据(当前为nullptr)
  - `device::ascend::Handle *handle_ascned`: 昇腾设备句柄，包含device类型和ID
- **Core Methods**:
  - `~Descriptor()`: 默认析构函数
  - `static create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, x_desc, pos_desc, sin_desc, cos_desc, infiniopRoPEAlgo_t algo)`: 工厂方法，执行以下逻辑：
    1. 类型转换: `reinterpret_cast<device::ascend::Handle *>(handle)`
    2. 调用 `RoPEInfo::createRoPEInfo(...)` 验证并提取元数据，返回 `Result<RoPEInfo>`
    3. 使用 `CHECK_RESULT(result)` 宏检查错误码，失败则提前返回
    4. 验证算法类型: `if (algo != INFINI_ROPE_ALGO_GPT_J) return INFINI_STATUS_NOT_IMPLEMENTED`
    5. 分配Descriptor实例: `new Descriptor(std::move(result.take()), 0, nullptr, handle_ascned->device, handle_ascned->device_id)`
  - `calculate(void *workspace, size_t workspace_size, void *y, const void *x, const void *pos_ids, const void *sin_table, const void *cos_table, void *stream) const`: 执行kernel启动，包含以下步骤：
    1. 数据类型校验: `CHECK_DTYPE(_info.data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F16)`
    2. 从 `_info` 提取所有kernel参数 (seq_len, nhead, dhead, strides等)
    3. 调用 `rope_kernel_launch(...)` 启动AI Core kernel
- **Lifecycle**:
  - 由 `infiniopCreateRopeDescriptor` (外部API) 创建
  - 用户负责调用 `infiniopDestroyDescriptor` 销毁

### `RoPEInfo`
- **Location**: `rope.h:56-204`
- **Primary Function**: RoPE操作的元数据容器，提供 `createRoPEInfo` 静态工厂方法验证张量形状兼容性并计算stride
- **Key Members**:
  - `infiniDtype_t data_type, pos_type`: 输入/位置ID的数据类型
  - `size_t batch, seqlen, nhead, dhead, table_len, table_dim`: 张量形状维度
  - `ptrdiff_t y_stride_batch, y_stride_seqlen, y_stride_nhead`: 输出张量的stride (支持非连续内存)
  - `ptrdiff_t x_stride_batch, x_stride_seqlen, x_stride_nhead`: 输入张量的stride
  - `bool has_batch_dim, pos_has_batch_dim`: 是否包含batch维度
  - `infiniopRoPEAlgo_t algo`: 算法变体 (当前仅支持GPT-J)
- **Core Methods**:
  - `static createRoPEInfo(infiniopTensorDescriptor_t y_desc, x_desc, pos_desc, sin_desc, cos_desc, infiniopRoPEAlgo_t algo)`: 验证并构建RoPEInfo，检查规则包括：
    1. 非空指针检查
    2. 数据类型一致性: `data_type == x_desc->dtype() == sin_desc->dtype() == cos_desc->dtype()`
    3. 支持的数据类型: `F16, BF16, F32, F64` (输入), `任何整数类型` (位置ID)
    4. 形状兼容性:
       - 3D张量: `[seqlen, nhead, dhead]`
       - 4D张量: `[batch, seqlen, nhead, dhead]`
       - sin/cos表: `[table_len, table_dim]` (必须连续)
       - 位置ID: 1D `[seqlen]` 或 2D `[batch, seqlen]`
    5. 约束: `dhead == table_dim * 2` (RoPE要求head维度是sin/cos表维度的2倍)
    6. 内存连续性: 最后一维stride必须为1，sin/cos表必须完全连续
    7. 提取stride信息，对于3D张量设置 `batch_stride=0`

## 3. API Interface

```cpp
// Kernel启动接口 (rope_ascend.h:6-21)
extern "C" infiniStatus_t rope_kernel_launch(
    void *y,                   // 输出张量 [seqlen, nhead, dhead] 或 [batch, seqlen, nhead, dhead]
    void *x,                   // 输入张量 (形状同y)
    void *pos,                 // 位置ID [seqlen] 或 [batch, seqlen]
    void *sin,                 // sin表 [table_len, table_dim] (连续内存)
    void *cos,                 // cos表 [table_len, table_dim] (连续内存)
    size_t seq_len,            // 序列长度
    size_t nhead,              // 注意力头数量
    size_t dhead,              // 每个头的维度
    infiniDtype_t data_type,   // 输入数据类型 (F16/F32)
    infiniDtype_t pos_type,    // 位置ID类型 (任意整数类型)
    ptrdiff_t y_stride_seqlen, // 输出stride (seqlen维度)
    ptrdiff_t y_stride_nhead,  // 输出stride (nhead维度)
    ptrdiff_t x_stride_seqlen, // 输入stride (seqlen维度)
    ptrdiff_t x_stride_nhead,  // 输入stride (nhead维度)
    void *stream               // Ascend ACL stream
);

// AI Core Kernel函数签名 (rope_ascend_kernel.cpp:222-235)
template <typename T>
__global__ __aicore__ void rope_kernel_float(
    GM_ADDR y, x, pos, sin, cos,        // Global Memory地址
    size_t seq_len, dhead,               // 序列长度和头维度
    ptrdiff_t y_stride_seqlen,           // 输出stride
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,           // 输入stride
    ptrdiff_t x_stride_nhead,
    int32_t pos_type                     // 位置ID数据类型枚举
);

// 半精度版本
template <typename T>
__global__ __aicore__ void rope_kernel_half(...);
```

## 4. Usage Example

```cpp
// 示例: 在华为昇腾NPU上执行RoPE操作 (GPT-J算法)
#include "infiniop.h"
#include "rope_ascend.h"

// 1. 创建张量描述符 (假设输入形状为 [batch=2, seqlen=128, nhead=32, dhead=128])
int64_t y_shape[4] = {2, 128, 32, 128};
int64_t y_stride[4] = {128*32*128, 32*128, 128, 1};  // 连续内存
infiniopTensorDescriptor_t y_desc;
infiniopCreateTensorDescriptor(&y_desc, kINFINI_DEVICE_ASCEND, 0,
                               INFINI_DTYPE_F16, 4, y_shape, y_stride);

// 输入张量x (形状同y)
infiniopTensorDescriptor_t x_desc;
infiniopCreateTensorDescriptor(&x_desc, kINFINI_DEVICE_ASCEND, 0,
                               INFINI_DTYPE_F16, 4, y_shape, y_stride);

// 位置ID [batch=2, seqlen=128]
int64_t pos_shape[2] = {2, 128};
int64_t pos_stride[2] = {128, 1};
infiniopTensorDescriptor_t pos_desc;
infiniopCreateTensorDescriptor(&pos_desc, kINFINI_DEVICE_ASCEND, 0,
                               INFINI_DTYPE_I32, 2, pos_shape, pos_stride);

// sin/cos表 [table_len=2048, table_dim=64] (注意: table_dim = dhead/2)
int64_t table_shape[2] = {2048, 64};
int64_t table_stride[2] = {64, 1};
infiniopTensorDescriptor_t sin_desc, cos_desc;
infiniopCreateTensorDescriptor(&sin_desc, kINFINI_DEVICE_ASCEND, 0,
                               INFINI_DTYPE_F16, 2, table_shape, table_stride);
infiniopCreateTensorDescriptor(&cos_desc, kINFINI_DEVICE_ASCEND, 0,
                               INFINI_DTYPE_F16, 2, table_shape, table_stride);

// 2. 创建RoPE描述符
infiniopHandle_t handle;  // 假设已初始化的Ascend句柄
infiniopRopeDescriptor_t rope_desc;
infiniStatus_t status = infiniopCreateRopeDescriptor(
    handle, &rope_desc,
    y_desc, x_desc, pos_desc, sin_desc, cos_desc,
    INFINI_ROPE_ALGO_GPT_J  // 仅支持GPT-J算法
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 分配设备内存并准备数据
void *d_x, *d_y, *d_pos, *d_sin, *d_cos;
size_t x_bytes = 2 * 128 * 32 * 128 * sizeof(half);  // batch*seqlen*nhead*dhead
size_t pos_bytes = 2 * 128 * sizeof(int32_t);
size_t table_bytes = 2048 * 64 * sizeof(half);

aclrtMalloc(&d_x, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_y, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_pos, pos_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_sin, table_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_cos, table_bytes, ACL_MEM_MALLOC_HUGE_FIRST);

// 拷贝数据到设备 (假设h_x, h_pos, h_sin, h_cos为主机端指针)
aclrtMemcpy(d_x, x_bytes, h_x, x_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(d_pos, pos_bytes, h_pos, pos_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(d_sin, table_bytes, h_sin, table_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(d_cos, table_bytes, h_cos, table_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

// 4. 创建ACL stream
aclrtStream stream;
aclrtCreateStream(&stream);

// 5. 执行RoPE计算
size_t workspace_size = infiniopGetRopeWorkspaceSize(rope_desc);
void *workspace = nullptr;  // 当前实现workspace_size=0
status = infiniopRope(
    rope_desc,
    workspace, workspace_size,
    d_y, d_x, d_pos, d_sin, d_cos,
    stream
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 6. 同步并取回结果
aclrtSynchronizeStream(stream);
half *h_y = new half[2 * 128 * 32 * 128];
aclrtMemcpy(h_y, x_bytes, d_y, x_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

// 7. 清理资源
aclrtDestroyStream(stream);
aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_pos);
aclrtFree(d_sin); aclrtFree(d_cos);
infiniopDestroyRopeDescriptor(rope_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(pos_desc);
infiniopDestroyTensorDescriptor(sin_desc);
infiniopDestroyTensorDescriptor(cos_desc);
```

## 5. Implementation Details

### 内存管理与数据流
- **分层存储架构**:
  - **Global Memory (GM)**: HBM/HBM，存储完整输入/输出张量、位置ID表、sin/cos表，通过 `GlobalTensor<T>` 指针访问
  - **Unified Buffer (UB)**: AI Core片上共享缓存，通过 `TQue` 管理的ping-pong缓冲区，容量有限但延迟低
  - **Vector Compute Buffer**: VECCALC区域的临时计算buffer (`_tmp_odd_buf`, `_tmp_even_buf`等)，用于奇偶分离和中间结果存储
- **双缓冲流水线**: 使用 `BUFFER_NUM=2` 实现ping-pong机制，`copyIn` 和 `copyOut` 可与 `compute` 重叠执行，隐藏数据搬运延迟
- **32字节对齐**: `alignTileLen<T>(dh, BYTE_ALIGN)` 确保每次DMA搬运长度满足32字节对齐要求(BYTE_ALIGN=32)，提升传输效率
- **Padding处理**: 输入搬运使用 `DataCopy` (可能包含padding)，输出搬运使用 `DataCopyPad` 明确处理padding区域

### 并行计算策略
- **Block并行**: 每个AI Core Block处理一个注意力头(head)，启动参数 `<<<nhead, nullptr, stream>>>` 决定并行度
- **Block索引**: 通过 `GetBlockIdx()` 获取当前Block ID (0到nhead-1)，用于计算当前处理的head在全局内存中的偏移
- **向量化计算**: 所有运算使用AscendC的向量API (`Mul`, `Sub`, `Add`, `GatherMask`)，单次指令处理 `_tile_len/2` 个元素
- **流水线并行**: `TPipe` 协调VECIN和VECOUT队列的同步，`copyIn`、`compute`、`copyOut` 形成三级流水线

### 算法实现细节
- **RoPE公式 (GPT-J变体)**:
  - 给定输入向量 `x` (维度dhead)，按奇偶索引拆分为 `x_odd` 和 `x_even`
  - 旋转后的输出: `y_odd = x_odd * cos - x_even * sin`, `y_even = x_odd * sin + x_even * cos`
  - 等价于2D旋转矩阵 `[cos -sin; sin cos]` 作用在 `(x_odd, x_even)` 上
- **索引计算**:
  - 输入索引: `idx = i * st_xnt + block_idx * st_xnh` (i=序列位置, block_idx=head)
  - 位置ID索引: `pos_idx = _p_gm(i)` (支持1D共享位置或2D每batch独立位置)
  - sin/cos索引: `pos_idx * tile_len/2` (每个位置对应 `dhead/2` 个sin/cos值)
- **GatherMask实现**:
  - `GatherMask<T>(dst, src, mask_mode, reverse, repeat, gMaskParams, rsvdCnt)`
  - `mask_mode=1`: 选择索引0,2,4... (奇索引元素)
  - `mask_mode=2`: 选择索引1,3,5... (偶索引元素)
  - `repeat_count=ceil((tile_len*sizeof(T))/256)`: 处理超过256字节的分片

### 性能优化技术
- **减少全局内存访问**: sin/cos表按需加载 (每个序列位置仅加载一次，复用于所有head)
- **计算密集型优化**: 奇偶分离后每个子向量长度为 `dhead/2`，8次向量乘法 (odd*cos, even*sin, odd*sin, even*cos) 充分利用AI Core的向量计算单元
- **同步开销最小化**: 使用 `PipeBarrier<PIPE_V>()` 仅在必要位置同步向量计算，避免过度同步
- **循环展开**: `copyOut` 中的奇偶合并循环由编译器自动优化，显式逐元素赋值确保正确性

### 错误处理与约束
- **数据类型检查**: 仅支持 `INFINI_DTYPE_F16` 和 `INFINI_DTYPE_F32`，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **算法限制**: 当前仅实现 `INFINI_ROPE_ALGO_GPT_J`，其他算法 (如Standard RoPE) 返回 `INFINI_STATUS_NOT_IMPLEMENTED`
- **形状验证**: `RoPEInfo::createRoPEInfo` 严格检查张量形状兼容性，不满足则返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **Stride约束**: 最后一维stride必须为1，sin/cos表必须完全连续，否则返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
- **设备管理**: 通过 `device::ascend::Handle` 获取device类型和ID，支持多卡环境

### 依赖关系
- **AscendC SDK**: 依赖Ascend AI Core编程模型 (`AscendC` 命名空间)，包括 `TPipe`, `TQue`, `TBuf`, `GlobalTensor`, `LocalTensor`, `DataCopy`, `GatherMask`, `Mul`, `Sub`, `Add` 等API
- **ACL (Ascend Computing Language)**: 通过 `aclrtMalloc`, `aclrtMemcpy`, `aclrtStream` 管理设备和内存
- **InfiniOp框架**: 继承 `InfiniopDescriptor` 基类，使用 `infiniopTensorDescriptor_t` 统一张量描述接口
- **通用工具**: 使用 `utils::Result<T>` 进行错误传播，`CHECK_RESULT`, `CHECK_DTYPE` 宏简化错误检查
- **常量定义**: `BLOCK_NUM=8`, `BUFFER_NUM=2`, `BYTE_ALIGN=32`, `BLOCK_LEN=256` (定义于 `ascend_kernel_common.h`)

### 设计模式
- **Factory Pattern**: `Descriptor::create` 静态工厂方法封装创建逻辑
- **Strategy Pattern**: 通过 `algo` 参数选择不同RoPE算法变体 (当前仅GPT-J)
- **Template Method**: `RoPEKernel` 模板类支持数据类型和位置ID类型的任意组合
- **RAII**: 使用 `TQue::AllocTensor` / `FreeTensor`, `LocalTensor` 的作用域管理自动释放资源
- **Macro-based Code Generation**: `DESCRIPTOR`, `CASE_POSTYPE`, `ROPE_KERNEL`, `DEFINE_ROPE_KERNEL` 等宏减少重复代码
