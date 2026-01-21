# SwiGLU KUNLUN Operator Implementation Documentation

SwiGLU (Swish-Gated Linear Unit) 算子昆仑(KUNLUN) XPU 设备实现，用于深度学习模型中的前馈神经网络层。该模块基于通用逐元素(elementwise)操作框架，提供高性能的 SwiGLU 激活函数计算，支持 FP32、FP16 和 BF16 数据类型，并针对昆仑 XPU 架构进行了优化。

## 1. 模块结构

- **`kernel.h`**: 核心 SwiGLU 操作算子定义，包含设备端 sigmoid 函数和主运算符 functor
- **`swiglu_kunlun.h`**: 公共 API 声明，通过 ELEMENTWISE_DESCRIPTOR 宏生成 Descriptor 类接口
- **`swiglu_kunlun.xpu`**: Descriptor 实现，包含算子创建(create)和计算(calculate)方法

## 2. 核心类

### `SwiGLUOp`
- **位置**: `kernel.h`
- **主要功能**: SwiGLU 激活函数的设备端运算符，实现 `Swish(x) * x` 的变体形式 `sigmoid(gate) * gate * up`
- **关键成员**:
  - `num_inputs`: 静态常量，值为 2，表示接收两个输入张量(up 和 gate)

- **核心方法**:
  - `sigmoid<T>(T x)`: 设备端 sigmoid 激活函数，使用 `1.0f / (1.0f + exp(-x))` 公式计算
  - `sigmoidf(float x)`: float 类型特化的 sigmoid 函数，提供与泛型版本相同实现
  - `operator()(const T *inputs)`: 主运算符重载，接收输入数组指针
    - 对于 FP32/FP16: 计算 `gate * sigmoid(gate) * up`，其中 `up = inputs[0]`, `gate = inputs[1]`
    - 对于 BF16: 先转换为 float 精度计算，再转回 BF16，避免精度损失

- **生命周期**: 值类型语义，无状态函数对象，每次调用独立执行

## 3. API 接口

```cpp
namespace op::swiglu::kunlun {

// 通过 ELEMENTWISE_DESCRIPTOR 宏自动生成的 Descriptor 类
class Descriptor final : public InfiniopDescriptor {
public:
    // 析构函数
    ~Descriptor();

    // 获取所需工作空间大小
    size_t workspaceSize() const;

    // 创建 SwiGLU 算子描述符
    static infiniStatus_t create(
        infiniopHandle_t handle,                  // 昆仑设备句柄
        Descriptor **desc_ptr,                    // 输出：创建的描述符指针
        infiniopTensorDescriptor_t out_desc,      // 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_descs); // 输入张量描述符向量

    // 执行 SwiGLU 计算
    infiniStatus_t calculate(
        void *workspace,              // 设备工作空间指针
        size_t workspace_size,        // 工作空间大小
        void *output,                 // 输出张量设备指针
        std::vector<const void *> inputs, // 输入张量设备指针向量
        void *stream) const;          // 昆仑流
};

} // namespace op::swiglu::kunlun
```

### SwiGLU 数学定义
```
SwiGLU(up, gate) = up * (gate * sigmoid(gate))
                 = up * gate * (1 / (1 + e^(-gate)))
```

## 4. 使用示例

```cpp
// 示例：在昆仑 XPU 上执行 SwiGLU 操作
#include "swiglu_kunlun.h"

using namespace op::swiglu::kunlun;

// 1. 准备张量描述符 (假设形状为 [batch_size, seq_len, hidden_dim])
std::vector<int64_t> shape = {32, 512, 4096};
std::vector<int64_t> strides = {512 * 4096, 4096, 1};

infiniopTensorDescriptor_t up_desc, gate_desc, out_desc;
// ... 创建张量描述符 (省略具体代码) ...

// 2. 创建算子描述符
Descriptor *swiglu_desc = nullptr;
infiniStatus_t status = Descriptor::create(
    kunlun_handle,        // 昆仑设备句柄
    &swiglu_desc,         // 输出描述符指针
    out_desc,             // 输出张量
    {up_desc, gate_desc}  // 输入张量向量
);

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配工作空间
size_t workspace_size = swiglu_desc->workspaceSize();
void *workspace;
xpu_malloc(workspace, workspace_size);

// 4. 执行计算
void *d_up, *d_gate, *d_output;
// ... 假设已在设备上分配输入输出内存 ...

kunlunStream_t stream;
xpu_stream_create(&stream);

status = swiglu_desc->calculate(
    workspace,          // 设备工作空间
    workspace_size,     // 工作空间大小
    d_output,           // 输出张量设备指针
    {d_up, d_gate},     // 输入张量设备指针向量
    stream              // 昆仑流
);

// 5. 同步和清理
xpu_stream_synchronize(stream);
xpu_free(workspace);
delete swiglu_desc;
```

## 5. 实现细节

### 内存管理
- **工作空间分配**: 使用 `ElementwiseInfo` 结构计算所需工作空间，大小为 `meta_mem_size + num_inputs * sizeof(void*)`
  - `meta_mem_size`: 存储 shape、stride、contiguous、broadcasted 等元数据
  - `num_inputs * sizeof(void*)`: 存储输入张量设备指针数组
- **设备内存传输**: 通过 `xpu_memcpy_async` 异步将元数据和输入指针从主机传输到设备
- **共享内存使用**: XPU kernel 使用 `__local__` 本地内存缓存 shape、stride 和 contiguous 标志

### 并发策略
- **Block 配置**: 固定使用 8 个 cluster，每个 cluster 64 个计算核心
- **线程调度**: 基于 `core_id()` 和 `cluster_id()` 的全局线程 ID 进行任务分配
  - `thread_id = ncores * cluster_id() + cid`
  - 总线程数 `nthreads = ncores * cluster_num()`
- **数据并行**: 输出张量被切分为多个块，每个线程处理 `BUFF_SIZE=64` 个元素
- **同步机制**: 使用 `mfence()` 确保本地内存加载完成后进行计算，使用 `sync_cluster()` 进行 cluster 级同步

### 性能优化
- **批量处理**: 每次循环处理最多 64 个元素(`BUFF_SIZE`)，减少全局内存访问次数
- **本地内存缓存**: shape、stride、contiguous、broadcasted 等元数据缓存在本地内存，避免重复全局访问
- **异步内存传输**: 使用 `GM2LM_ASYNC` 和 `LM2GM_ASYNC` 异步传输全局内存和本地内存数据
- **SIMD 优化**: 昆仑 XPU 的 512 位寄存器可用于向量化计算(由 XPU 编译器自动优化)
- **分支消除**: 使用模板特化和 `constexpr if` 编译期分支，避免运行时条件判断

### 错误处理
- **数据类型检查**: 支持 FP32、FP16、BF16，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状一致性**: 使用 `CHECK_SAME_SHAPE` 宏验证输出和两个输入张量形状完全一致
- **工作空间验证**: `calculate` 方法检查 `workspace_size` 是否满足要求，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **空张量处理**: `ElementwiseInfo::create` 检查空指针，返回 `INFINI_STATUS_BAD_PARAM`
- **广播限制**: 输出张量不允许有广播维度(`hasBroadcastDim()`)，返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`

### 依赖关系
- **内部依赖**:
  - `/infiniop/elementwise/elementwise.h`: 通用逐元素操作框架
  - `/infiniop/elementwise/kunlun/elementwise_kunlun.h`: 昆仑设备逐元素操作实现
  - `/infiniop/devices/kunlun/kunlun_common.h`: 昆仑设备类型定义和宏
  - `/infiniop/devices/kunlun/kunlun_kernel_common.h`: 昆仑 kernel 通用工具(indexToOffset, atomicAdd 等)
- **外部依赖**:
  - `xpu/runtime.h`: 昆仑 XPU 运行时 API
  - `xpu/kernel/xtdk.h`: XPU 内核开发工具包
  - `xpu/kernel/xtdk_bf16.h`: BF16 数学函数支持
  - `xpu/kernel/xtdk_math.h`: 数学函数(exp, sqrt 等)

### 设计模式
- **Functor 模式**: `SwiGLUOp` 作为函数对象，通过 `operator()` 实现操作符语义
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR` 宏生成派生自 `InfiniopDescriptor` 的专用类
- **策略模式**: 通过模板参数 `Op` 将具体操作注入通用逐元素 kernel 框架
- **Pimpl 模式**: `DeviceImpl::Opaque` 隐藏设备实现细节
- **RAII**: 使用 `std::unique_ptr` 管理 `DeviceImpl` 生命周期
- **模板特化**: 为 BF16 类型提供特化版本，在 float 精度下计算以保证数值稳定性

### XPU Kernel 执行流程
1. **元数据加载**: 通过 `GM2LM_ASYNC` 将 shape、stride、contiguous 标志加载到本地内存
2. **内存屏障**: `mfence()` 确保所有元数据加载完成
3. **循环分块**: 每个线程处理 `len_per_loop = min(64, ceil_div(output_size, nthreads))` 个元素
4. **索引计算**:
   - 输出索引: `getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides)`
   - 输入索引: `InputIndexer` 根据 contiguous/broadcasted 标志计算每个输入的内存偏移
5. **数据搬运**: 将输入数据从全局内存异步加载到本地内存缓冲区
6. **操作执行**: 调用 `SwiGLUOp{}(inputs_buf)` 计算结果
7. **结果写回**: 将结果异步从本地内存写回到全局内存
8. **集群同步**: `sync_cluster()` 确保所有 cluster 完成计算

### 数值精度处理
- **BF16 特化**: 使用 `__bfloat162float` 和 `__float2bfloat16` 进行精度转换
  - 计算时提升到 float 避免下溢和精度损失
  - sigmoid 和乘法都在 float 域执行
- **FP16 通用**: 使用模板泛型实现，由 XPU 硬件自动处理 FP16 运算
- **FP32 原生**: 直接使用 float 类型，无需转换
