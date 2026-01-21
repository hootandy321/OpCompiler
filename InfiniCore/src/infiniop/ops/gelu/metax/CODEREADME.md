# GELU METAX 核心实现文档

本模块实现了 GELU (Gaussian Error Linear Unit) 激活函数在 Metax 设备上的后端，基于统一的逐元素操作框架，通过复用 CUDA 核算逻辑实现了高效的单目算子运算。

## 1. 模块结构

- **`gelu_metax.h`**: 元数据描述符定义，使用宏生成统一的 Descriptor 类接口
- **`gelu_meta.maca`**: Metax 后端实现核心，包含描述符创建与计算调度逻辑

## 2. 核心类

### `op::gelu::metax::Descriptor`
- **位置**: `gelu_metax.h` (通过 ELEMENTWISE_DESCRIPTOR 宏生成)
- **主要功能**: GELU 算子的 Metax 设备描述符，负责类型检查、形状验证和计算调度
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 支持的数据类型 (BF16/F16/F32/F64)
  - `_info`: `op::elementwise::ElementwiseInfo` - 张量形状、步幅、广播等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::metax::DeviceImpl>` - Metax 设备实现实例
  - `_workspace_size`: `size_t` - 设备端所需工作空间大小
- **核心方法**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 构造描述符，验证输入输出张量形状一致性，创建 ElementwiseInfo 元数据，初始化 Metax 设备实现
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 GELU 计算，根据数据类型分派到对应的模板特化 (cuda_bfloat16/half/float/double)
- **生命周期**: 通过 `create` 静态方法构造，析构时由基类管理资源，使用 RAII 模式自动清理设备实现对象

### `op::gelu::cuda::GeluOp`
- **位置**: `../cuda/kernel.cuh` (被 Metax 后端复用)
- **主要功能**: GELU 操作的 CUDA 设备函数对象，实现数学公式 `0.5 * x * (1 + erf(x / sqrt(2.0)))`
- **关键成员**:
  - `num_inputs`: `constexpr size_t = 1` - 单目操作符标识
- **核心方法**:
  - `operator()(const T &x)`: 设备端仿函数，根据类型特化执行不同精度计算
    - BF16: 转换为 float 计算，再转回 bfloat16 (`__bfloat162float` / `__float2bfloat16`)
    - FP16: 转换为 float 计算，再转回 half (`__half2float` / `__float2half`)
    - FP32/FP64: 直接使用 `erf` 标准库函数计算
- **设计模式**: Functor 模式，支持编译期类型分发 (`if constexpr`)

### `op::elementwise::metax::DeviceImpl`
- **位置**: `../../elementwise/metax/elementwise_metax.h`
- **主要功能**: Metax 设备端通用逐元素操作执行引擎，负责内核启动、内存传输、网格配置
- **关键成员**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl 模式隐藏实现细节
- **核心方法**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(info, workspace, output, inputs, stream)`: 单一数据类型的计算入口，分发到 `calculateImpl`
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(...)`: 调用 `launchElementwiseKernel` 启动 Metax 内核
  - `launchElementwiseKernel(...)`: 计算网格维度，循环启动内核处理大规模张量 (单次网格最大处理 `gridDims.x * blockDims.x` 个元素)
  - `infoToDevice<N>(...)`: 将元数据 (形状/步幅/广播标志) 从主机异步拷贝到设备工作空间
- **生命周期**: 通过 `create` 工厂方法构造，使用共享指针管理 Opaque 实例

## 3. API 接口

```cpp
// 创建 GELU Metax 描述符
namespace op::gelu::metax {
    class Descriptor : public InfiniopDescriptor {
    public:
        ~Descriptor();

        // 构造描述符并验证参数
        static infiniStatus_t create(
            infiniopHandle_t handle,                  // Metax 设备句柄
            Descriptor **desc_ptr,                    // 输出描述符指针
            infiniopTensorDescriptor_t output_desc,   // 输出张量描述符
            std::vector<infiniopTensorDescriptor_t> input_descs); // 单个输入张量

        // 执行 GELU 计算
        infiniStatus_t calculate(
            void *workspace,               // 设备工作空间指针
            size_t workspace_size,         // 工作空间大小 (>= _workspace_size)
            void *output,                  // 输出张量设备指针
            std::vector<const void *> inputs, // 输入张量设备指针数组 (单元素)
            void *stream) const;           // Metax 计算流
    };
}

// CUDA 设备端操作符 (被 Metax 后端复用)
namespace op::gelu::cuda {
    struct GeluOp {
        static constexpr size_t num_inputs = 1;

        template <typename T>
        __device__ __forceinline__ T operator()(const T &x) const;
    };
}

// Metax 逐元素操作设备实现
namespace op::elementwise::metax {
    class DeviceImpl {
    public:
        // 单一数据类型计算 (输入输出同类型)
        template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
        infiniStatus_t calculate(
            const op::elementwise::ElementwiseInfo &info,
            void *workspace,
            void *output,
            const std::vector<const void *> &inputs,
            void *stream,
            Args &&...args);

        // 多类型计算 (支持输入输出类型不同)
        template <uint32_t BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args>
        infiniStatus_t calculate(
            const op::elementwise::ElementwiseInfo &info,
            void *workspace,
            void *output,
            const std::vector<const void *> &inputs,
            void *stream,
            Args &&...args);
    };
}
```

## 4. 使用示例

```cpp
// 示例: 在 Metax 设备上执行 GELU 激活
#include "gelu_metax.h"

// 1. 创建 Metax 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, device_id);

// 2. 准备张量描述符 (假设输入形状为 [1024, 1024])
std::vector<size_t> shape = {1024, 1024};
std::vector<ptrdiff_t> strides = {1024, 1}; // C 连续内存

infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, shape, strides);

// 3. 创建 GELU 描述符
op::gelu::metax::Descriptor *gelu_desc;
auto status = op::gelu::metax::Descriptor::create(
    handle, &gelu_desc, output_desc, {input_desc});

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误 (类型不匹配/形状不一致等)
}

// 4. 分配工作空间
size_t workspace_size = gelu_desc->workspaceSize();
void *workspace;
hcMalloc(&workspace, workspace_size);

// 5. 分配输入输出设备内存
float *d_input, *d_output;
hcMalloc((void**)&d_input, 1024 * 1024 * sizeof(float));
hcMalloc((void**)&d_output, 1024 * 1024 * sizeof(float));

// 6. 拷贝输入数据到设备
hcMemcpyAsync(d_input, h_input, 1024 * 1024 * sizeof(float),
              hcMemcpyHostToDevice, stream);

// 7. 执行 GELU 计算
status = gelu_desc->calculate(workspace, workspace_size,
                              d_output, {d_input}, stream);

// 8. 拷贝结果回主机
hcMemcpyAsync(h_output, d_output, 1024 * 1024 * sizeof(float),
              hcMemcpyDeviceToHost, stream);
hcStreamSynchronize(stream);

// 9. 清理资源
hcFree(d_input);
hcFree(d_output);
hcFree(workspace);
delete gelu_desc;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **工作空间布局**: 分配 `input_size * sizeof(void*)` 字节存储输入指针数组，其后紧接 `ElementwiseInfo.getMetaMemSize()` 字节存储元数据
- **元数据传输**: 通过 `hcMemcpyAsync` 将形状、步幅、连续性标志批量传输到设备，避免多次小拷贝开销
- **Pimpl 模式**: `DeviceImpl` 使用 `Opaque` 结构体隐藏设备句柄和内核实现细节，减少编译依赖

### 并发机制
- **流式执行**: 所有计算在用户提供的 `hcStream_t` 上异步执行，支持多流并行
- **网格循环**: 对于超大张量 (> `gridSizeX * maxThreadsPerBlock`)，采用循环启动内核策略，每次处理一个网格步长 (`step = gridDims.x * blockDims.x`)
- **无同步点**: 计算内核与主机异步执行，用户需显式调用同步函数

### 性能优化
- **块大小**: 固定使用 `BLOCK_SIZE = 256` 线程块，平衡寄存器占用与warp调度效率
- **连续内存优化**: 通过 `isOutputContiguous()` 和 `input_contiguous[]` 标志跳过索引计算，直接使用线性索引 (`idx`)
- **类型特化**: 编译期为 BF16/F16/F32/F64 生成专用内核，避免运行时分支
- **编译期展开**: 使用 `std::make_index_sequence<N>` 在编译期展开输入访问循环 (GELU 的 N=1，无实际展开成本)

### 错误处理
- **类型检查**: `CHECK_DTYPE` 宏验证仅支持 BF16/F16/F32/F64 四种精度
- **形状验证**: `CHECK_SAME_SHAPE` 确保输入输出张量形状完全一致
- **工作空间**: `calculate` 方法检查 `workspace_size < _workspace_size` 并返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **设备通信**: `CHECK_METAX` 和 `CHECK_STATUS` 宏捕获所有 Metax API 错误码

### 依赖关系
- **CUDA 互操作**: 复用 `../cuda/kernel.cuh` 中的 `GeluOp`，Metax 后端直接调用 CUDA 设备函数 (依赖 Metax 对 CUDA 的兼容层)
- **逐元素框架**: 继承 `op::elementwise::metax::DeviceImpl` 的通用内核启动逻辑
- **元数据结构**: 使用 `op::elementwise::ElementwiseInfo` 统一描述张量布局

### 设计模式
- **宏生成**: `ELEMENTWISE_DESCRIPTOR(gelu, metax)` 宏展开为完整的 `Descriptor` 类定义，减少重复代码
- **CRTP (Curiously Recurring Template Pattern)**: `GeluOp` 作为模板参数传递给 `calculate`，实现编译期多态
- **策略模式**: `calculate` 方法根据 `_dtype` 动态选择模板特化，实现运行时分发
- **工厂方法**: `Descriptor::create` 和 `DeviceImpl::create` 封装对象构造逻辑
