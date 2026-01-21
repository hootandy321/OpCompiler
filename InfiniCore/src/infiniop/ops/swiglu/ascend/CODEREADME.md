# SwiGLU Ascend 算子核心实现文档

华为昇腾(Ascend) AI 处理器上的 SwiGLU 激活函数算子实现。SwiGLU 是现代 Transformer 架构(如 LLaMA)中广泛使用的门控线性单元激活函数，计算公式为 `SwiGLU(x) = (x * W) ⊙ SiLU(x * V)`，其中 `SiLU(x) = x * sigmoid(x)`。本实现基于 Ascend C API，采用流水线并行和数据分块策略优化 NPU 计算性能。

## 1. 模块结构

- **`swiglu_ascend.h`**: 算子描述符与元数据定义，包含 `SwigluInfo` 验证类和 `Descriptor` 接口类
- **`swiglu_ascend.cc`**: 算子工厂实现，负责描述符创建与计算调度逻辑
- **`swiglu_ascend_kernel.cpp`**: Ascend C NPU 内核实现，包含流水线化的数据搬运与计算逻辑

## 2. 核心类

### `SwigluInfo`
- **位置**: `swiglu_ascend.h:10-48`
- **主要功能**: 张量描述符验证工厂类，执行严格的形状、步长、数据类型校验
- **关键成员**:
  - `dtype`: `infiniDtype_t` - 支持的数据类型(FP16/FP32)
  - `shape`: `std::vector<size_t>` - 张量维度，支持 2D `[seq, hidden]` 或 3D `[batch, seq, hidden]`
  - `ndim`: `int32_t` - 维度数(2 或 3)
  - `c_strides/a_strides/b_strides`: `std::vector<ptrdiff_t>` - 输出/输入张量的步长信息
- **核心方法**:
  - `create(c_desc, a_desc, b_desc)`: 静态工厂方法，返回 `Result<SwigluInfo>`
    - 验证所有张量非空: `CHECK_OR_RETURN(c_desc && a_desc && b_desc, BAD_PARAM)`
    - 禁止广播维度: `!c_desc->hasBroadcastDim()`
    - 维度必须是 2D 或 3D: `c_desc->ndim() == 2 || c_desc->ndim() == 3`
    - 三个张量形状完全一致: `CHECK_SAME_SHAPE(c_desc->shape(), a_desc->shape(), b_desc->shape())`
    - 最内层维度连续存储: `stride(ndim - 1) == 1`
    - 数据类型一致性: `c_desc->dtype() == a_desc->dtype() && c_desc->dtype() == b_desc->dtype()`
- **生命周期**: 值类型，通过 `create()` 静态方法构造，失败返回详细错误码

### `Descriptor`
- **位置**: `swiglu_ascend.h:50-70`
- **主要功能**: 算子描述符，继承自 `InfiniopDescriptor`，管理算子元数据与工作空间
- **关键成员**:
  - `_info`: `SwigluInfo` - 经过验证的张量元信息
  - `_workspace_size`: `size_t` - 工作空间大小(当前固定为 0，AscendC SwiGLU API 无需额外内存)
- **核心方法**:
  - `create(handle, desc_ptr, c_desc, input_descs)`: 静态工厂方法
    - 提取设备句柄: `reinterpret_cast<device::ascend::Handle *>(handle)`
    - 校验数据类型: `CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32)`
    - 调用 `SwigluInfo::create()` 验证张量描述符
    - 构造描述符: `new Descriptor(std::move(info), workspace_size, ...)`
  - `calculate(workspace, workspace_size, c, inputs, stream)`: 执行计算
    - 动态计算 batch/seq 维度大小: 2D 时 `batch=1, seq=shape[0]`，3D 时 `batch=shape[0], seq=shape[1]`
    - 提取 batch 和 seq 级别的步长(处理 2D/3D 兼容性)
    - 调用 `swiglu_kernel_launch()` 启动 NPU 内核
  - `workspaceSize()`: 返回所需工作空间大小(当前为 0)
- **生命周期**: 堆分配，由 `create()` 构造，使用者负责析构

### `SwigluKernel<T>`
- **位置**: `swiglu_ascend_kernel.cpp:6-131`
- **主要功能**: Ascend C NPU 内核模板类，实现流水线并行的 SwiGLU 计算
- **关键成员**:
  - `_c_gm/_a_gm/_b_gm`: `GlobalTensor<T>` - 全局内存(GM)中的输出/输入张量视图
  - `_in_queue_a/_in_queue_b`: `TQue<QuePosition::VECIN, BUFFER_NUM>` - 输入队列(双缓冲)
  - `_out_queue_c`: `TQue<QuePosition::VECOUT, BUFFER_NUM>` - 输出队列
  - `_pipe`: `TPipe` - 流水线管理器，负责队列内存分配
  - `_beta_value`: `float` - SwiGLU 的 β 参数(固定为 1.0)
  - `_block_idx/_tile_len/_copy_len`: `size_t` - 分块并行参数
  - `_batch/_seq_len/_hidden_size`: `size_t` - 张量形状维度
  - `_stride_seq_a/_stride_seq_b/_stride_seq_c`: `size_t` - 序列维度的步长
  - `_stride_batch_a/_stride_batch_b/_stride_batch_c`: `int64_t` - batch 维度的步长
- **核心方法**:
  - `init(c, a, b, batch_, seq, hd, ...)`: 初始化内核
    - 保存形状与步长参数到成员变量
    - 计算当前 NPU block 的分块长度: `_tile_len = (_block_idx < _hidden_size % BLOCK_NUM) ? (_hidden_size / BLOCK_NUM) + 1 : (_hidden_size / BLOCK_NUM)`
    - 对齐到字节边界: `_copy_len = alignTileLen<T>(_tile_len, BYTE_ALIGN)`
    - 绑定全局内存地址: `_a_gm.SetGlobalBuffer((__gm__ T *)a)`
    - 分配队列内存: `_pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T))`
  - `process()`: 主处理循环
    - 遍历所有 batch×seq 位置: `for (size_t i = 0; i < _batch * _seq_len; ++i)`
    - 流水线执行: `copyIn(i)` → `compute(i)` → `copyOut(i)`
  - `copyIn(i)`: 数据搬运(GM → Local)
    - 从队列分配 LocalTensor: `LocalTensor<T> aLocal = _in_queue_a.AllocTensor<T>()`
    - 计算当前 batch/seq 索引: `batch_idx = i / _seq_len`, `seq_idx = i % _seq_len`
    - 计算全局内存偏移: `idxa = batch_idx * _stride_batch_a + seq_idx * _stride_seq_a + _block_idx * _tile_len`
    - 异步数据搬运: `DataCopy(aLocal, _a_gm[idxa], _copy_len)`
    - 入队供计算使用: `_in_queue_a.EnQue(aLocal)`
  - `compute(i)`: SwiGLU 计算(Vector 计算)
    - 出队输入张量: `LocalTensor<T> aLocal = _in_queue_a.DeQue<T>()`
    - 分配输出张量: `LocalTensor<T> cLocal = _out_queue_c.AllocTensor<T>()`
    - 调用 AscendC API: `SwiGLU<T, false>(cLocal, aLocal, bLocal, _beta_value, _copy_len)`
    - 入队输出张量: `_out_queue_c.EnQue<T>(cLocal)`
    - 释放输入张量: `_in_queue_a.FreeTensor(aLocal)`
  - `copyOut(i)`: 数据搬运(Local → GM)
    - 出队输出张量: `LocalTensor<T> cLocal = _out_queue_c.DeQue<T>()`
    - 计算输出全局偏移: `idxc = batch_idx * _stride_batch_c + seq_idx * _stride_seq_c + _block_idx * _tile_len`
    - 处理非对齐尾部(使用 `DataCopyPad`): `if (_tile_len * sizeof(T) % BYTE_ALIGN != 0)`
    - 标准对齐数据(使用 `DataCopy`): `DataCopy(_c_gm[idxc], cLocal, _tile_len)`
    - 释放 LocalTensor: `_out_queue_c.FreeTensor(cLocal)`
- **生命周期**: NPU 上的栈分配对象，由内核启动时构造，计算完成后析构

## 3. API 接口

```cpp
// 算子描述符创建
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // Ascend 设备句柄
    Descriptor **desc_ptr,                            // 输出: 描述符指针
    infiniopTensorDescriptor_t c_desc,                // 输出张量描述符 [batch, seq, hidden] 或 [seq, hidden]
    std::vector<infiniopTensorDescriptor_t> input_descs // 输入张量描述符数组: {a_desc, b_desc}
);
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_PARAM / INFINI_STATUS_BAD_TENSOR_SHAPE / INFINI_STATUS_BAD_TENSOR_STRIDES / INFINI_STATUS_BAD_TENSOR_DTYPE

// 算子计算执行
infiniStatus_t Descriptor::calculate(
    void *workspace,              // 工作空间指针(当前未使用)
    size_t workspace_size,        // 工作空间大小(必须为 0)
    void *c,                      // 输出张量 GM 地址
    std::vector<const void *> inputs, // 输入张量 GM 地址数组: {a, b}
    void *stream                  // Ascend ACL stream 句柄
) const;
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE

// NPU 内核启动(C 接口)
extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,           // GM 地址
    infiniDtype_t dtype,                  // 数据类型(INFINI_DTYPE_F16/INFINI_DTYPE_F32)
    size_t batch, size_t seq, size_t hd,  // 张量维度
    ptrdiff_t stride_batch_c,             // batch 维度步长
    ptrdiff_t stride_batch_a,             // a 的 batch 维度步长
    ptrdiff_t stride_batch_b,             // b 的 batch 维度步长
    ptrdiff_t stride_seq_c,               // seq 维度步长
    ptrdiff_t stride_seq_a,               // a 的 seq 维度步长
    ptrdiff_t stride_seq_b,               // b 的 seq 维度步长
    void *stream                          // ACL stream
);
```

## 4. 使用示例

```cpp
// 示例: 在 Ascend NPU 上执行 SwiGLU 激活函数(3D 张量)
#include "swiglu_ascend.h"
#include "devices/ascend/common_ascend.h"

using namespace op::swiglu::ascend;

// 1. 创建 Ascend 设备句柄
device::ascend::Handle *handle;
// ... (假设已初始化)

// 2. 准备张量描述符 (batch=2, seq=1024, hidden=4096)
std::vector<size_t> shape = {2, 1024, 4096};
std::vector<ptrdiff_t> strides = {1024 * 4096, 4096, 1}; // C 连续

infiniopTensorDescriptor_t c_desc = /* 输出张量描述符, dtype=FP16 */;
infiniopTensorDescriptor_t a_desc = /* 输入张量 A 描述符, dtype=FP16 */;
infiniopTensorDescriptor_t b_desc = /* 输入张量 B 描述符, dtype=FP16 */;

// 3. 创建算子描述符
Descriptor *swiglu_desc;
infiniStatus_t status = Descriptor::create(
    handle,
    &swiglu_desc,
    c_desc,
    {a_desc, b_desc}
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误(形状不匹配、步长不连续、数据类型不支持等)
}

// 4. 分配 GM 内存
void *c_ptr, *a_ptr, *b_ptr;
aclrtMalloc(&c_ptr, c_desc->size(), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&a_ptr, a_desc->size(), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&b_ptr, b_desc->size(), ACL_MEM_MALLOC_HUGE_FIRST);

// 5. 准备 ACL stream
aclrtStream stream;
aclrtCreateStream(&stream);

// 6. 执行计算
size_t workspace_size = swiglu_desc->workspaceSize(); // 返回 0
void *workspace = nullptr; // 无需分配工作空间

status = swiglu_desc->calculate(
    workspace,
    workspace_size,
    c_ptr,          // 输出: SwiGLU(a, b)
    {a_ptr, b_ptr}, // 输入
    stream
);

// 7. 同步与清理
aclrtSynchronizeStream(stream);
aclrtDestroyStream(stream);

aclrtFree(c_ptr);
aclrtFree(a_ptr);
aclrtFree(b_ptr);

delete swiglu_desc;
```

```cpp
// 示例: 2D 张量 SwiGLU 计算(无 batch 维度)
std::vector<size_t> shape = {2048, 4096}; // [seq, hidden]
std::vector<ptrdiff_t> strides = {4096, 1};

// 创建算子描述符(内部会将 batch 视为 1)
Descriptor *swiglu_desc;
Descriptor::create(handle, &swiglu_desc, c_desc, {a_desc, b_desc});

// 执行计算(内部自动计算 batch=1, seq=2048)
swiglu_desc->calculate(nullptr, 0, c_ptr, {a_ptr, b_ptr}, stream);
```

## 5. 实现细节

- **内存管理**:
  - **零拷贝工作空间**: AscendC `SwiGLU` API 无需额外工作空间，`_workspace_size = 0`
  - **全局内存(GM)**: 张量数据存储在 HBM(High Bandwidth Memory)，通过 `GlobalTensor` 访问
  - **本地内存(Local)**: 每个 NPU core 的专用片上缓存，通过 `LocalTensor` 和队列管理
  - **队列缓冲**: `BUFFER_NUM` 个缓冲区的双缓冲/三缓冲机制，隐藏数据搬运延迟

- **并发与并行**:
  - **Block 级并行**: hidden 维度被切分为 `BLOCK_NUM` 个 block(通常等于 NPU core 数)，每个 block 处理 `_hidden_size / BLOCK_NUM` 个元素
  - **负载均衡**: 使用余数分配处理非均匀切分: `_block_idx < (_hidden_size % BLOCK_NUM) ? (_tile_len + 1) : _tile_len`
  - **流水线并行**: `copyIn` → `compute` → `copyOut` 三阶段流水线，数据搬运与计算重叠执行
  - **Stream 并发**: 支持多 stream 并发执行，通过 `aclrtStream` 参数传递

- **性能优化**:
  - **对齐策略**: 分块长度对齐到 `BYTE_ALIGN` 字节边界(通常 32 字节)，利用向量化指令
  - **批量处理**: 每次处理整个 `_tile_len` 长度的向量，而非逐元素循环
  - **非对齐尾部处理**: 使用 `DataCopyPad` 处理最后一个非对齐块，避免越界访问
  - **步长计算优化**: 2D/3D 统一处理，通过条件表达式动态计算步长，避免代码分支

- **错误处理**:
  - **编译期验证**: `SwigluInfo::create()` 在算子创建时执行严格检查，失败返回详细错误码
  - **运行时分发**: `swiglu_kernel_launch()` 根据 dtype 分发到专用内核(不支持的类型返回 `BAD_TENSOR_DTYPE`)
  - **错误传播**: 使用 `CHECK_OR_RETURN`, `CHECK_RESULT`, `CHECK_DTYPE` 宏进行错误传播

- **依赖项**:
  - **AscendC API**: `SwiGLU<T, isBias>()` - 华为昇腾向量计算 API
  - **AscendC 基础设施**: `GlobalTensor`, `LocalTensor`, `TQue`, `TPipe`, `DataCopy`, `DataCopyPad`
  - **Infini 框架**: `InfiniopDescriptor`, `infiniopTensorDescriptor_t`, `device::ascend::Handle`
  - **CANN 工具链**: ACL(Ascend Computing Language) runtime, `<<<BLOCK_NUM, 0, stream>>>` 内核启动语法

- **设计模式**:
  - **工厂模式**: `SwigluInfo::create()` 和 `Descriptor::create()` 静态工厂方法封装创建逻辑
  - **策略模式**: 模板参数 `T` 支持 FP16/FP32 两种数据类型策略
  - **RAII 惯例**: 队列内存通过 `AllocTensor`/`FreeTensor` 管理生命周期
  - **CRTP 模式**: `Descriptor` 继承基类 `InfiniopDescriptor`，实现多态算子接口
