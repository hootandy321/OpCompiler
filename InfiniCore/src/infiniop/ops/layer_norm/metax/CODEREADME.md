# Metax Layer Normalization Core Implementation Documentation

Metax硬件后端的Layer Normalization操作实现，为Infini框架提供基于Moore Threads GPU的层归一化计算功能。该模块通过复用CUDA实现并适配Metax特定的API（mcblas/mcdnn或hcblas/hcdnn），支持FP16、FP32和BF16数据类型的层归一化运算。

## 1. 模块结构

- **`layer_norm_metax.h`**: Metax后端Descriptor类的声明，使用宏定义DESCRIPTOR生成标准接口
- **`layer_norm_metax.maca`**: Metax后端的核心实现，包含内存管理、kernel启动和设备端计算逻辑

## 2. 核心类

### `Descriptor`
- **位置**: `layer_norm_metax.h` (通过DESCRIPTOR宏生成)
- **主要功能**: 封装Metax设备上的Layer Norm操作描述符，管理设备特定的资源、workspace大小和计算配置
- **关键成员**:
  - `_opaque`: 指向`Opaque`结构体的指针，包含`device::metax::Handle::Internal`共享指针用于访问设备属性
  - `_info`: `LayerNormInfo`实例，存储张量形状、步长、归一化维度等元数据
  - `_workspace_size`: workspace所需字节数，用于存储设备端步长数组
- **核心方法**:
  - `create(handle, desc_ptr, output_desc, input_standardization_desc, input_std_deviation_desc, input_desc, weight_desc, bias_desc, eps)`: 静态工厂方法，验证数据类型（F16/F32/BF16），创建`LayerNormInfo`，计算workspace大小（`sizeof(ptrdiff_t) * ndim * 4`），初始化Descriptor实例
  - `calculate(workspace, workspace_size, output, input_standardization, input_std_deviation, input, weight, bias, stream)`: 执行层归一化计算，将主机端步长异步拷贝到设备，根据设备maxThreadsPerBlock属性（1024或512）选择对应的BLOCK_SIZE，启动kernel
- **生命周期**: 由`create()`静态方法构造，析构函数释放`_opaque`资源；`Opaque`内部持有`device::metax::Handle::Internal`的共享指针确保设备句柄生命周期

### `Opaque`
- **位置**: `layer_norm_metax.maca` (Descriptor的内部结构体)
- **主要功能**: 封装Metax设备句柄的内部表示，提供设备架构查询能力
- **关键成员**:
  - `internal`: `std::shared_ptr<device::metax::Handle::Internal>`，通过`maxThreadsPerBlock()`查询设备支持的最大线程块大小（512或1024）
- **生命周期**: 在Descriptor创建时构造，随Descriptor析构而销毁；共享指针管理确保Handle的Internal对象存活

## 3. API接口

```cpp
// 创建Metax Layer Norm描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                                     // Metax设备句柄
    Descriptor **desc_ptr,                                       // 输出：创建的描述符指针
    infiniopTensorDescriptor_t output_desc,                      // 输出张量描述符
    infiniopTensorDescriptor_t input_standardization_desc,       // 标准化输出张量描述符
    infiniopTensorDescriptor_t input_std_deviation_desc,         // 标准差输出张量描述符
    infiniopTensorDescriptor_t input_desc,                       // 输入张量描述符
    infiniopTensorDescriptor_t weight_desc,                      // 权重张量描述符（1D，大小为normalized_size）
    infiniopTensorDescriptor_t bias_desc,                        // 偏置张量描述符（可选，可为nullptr）
    float eps                                                     // 防止除零的小常数
);
// 返回INFINI_STATUS_SUCCESS或错误码（数据类型不支持、形状不匹配等）

// 执行层归一化计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                                             // 设备端workspace指针
    size_t workspace_size,                                       // workspace大小（必须>=_workspace_size）
    void *output,                                                // 输出数据指针
    void *input_standardization,                                 // 标准化输出数据指针
    void *input_std_deviation,                                   // 标准差输出数据指针
    const void *input,                                           // 输入数据指针
    const void *weight,                                          // 权重数据指针
    const void *bias,                                            // 偏置数据指针（可选）
    void *stream                                                 // hcStream_t流
) const;
// 返回INFINI_STATUS_SUCCESS或INFINI_STATUS_INSUFFICIENT_WORKSPACE
```

## 4. 使用示例

```cpp
// 示例：在Metax GPU上执行Layer Normalization
#include "infiniop/ops/layer_norm/metax/layer_norm_metax.h"

// 1. 创建设备句柄
device::metax::Handle *handle = new device::metax::Handle(device_id);

// 2. 准备张量描述符（假设输入shape为[batch_size, seq_len, hidden_dim]）
std::vector<size_t> input_shape = {32, 128, 768};
std::vector<size_t> weight_shape = {768};  // 沿最后一个维度归一化

infiniopTensorDescriptor_t input_desc, output_desc, weight_desc, bias_desc;
infiniopTensorDescriptor_t input_standardization_desc, input_std_deviation_desc;

// ... (创建描述符的代码省略)

// 3. 创建Layer Norm描述符
op::layer_norm::metax::Descriptor *layer_norm_desc = nullptr;
float eps = 1e-5f;
infiniStatus_t status = op::layer_norm::metax::Descriptor::create(
    handle,
    &layer_norm_desc,
    output_desc,
    input_standardization_desc,
    input_std_deviation_desc,
    input_desc,
    weight_desc,
    bias_desc,  // 可选传nullptr
    eps
);

// 4. 分配workspace和设备内存
size_t workspace_size = layer_norm_desc->workspaceSize();
void *d_workspace;
hcMalloc(&d_workspace, workspace_size);

void *d_input, *d_output, *d_weight, *d_bias;
void *d_input_standardization, *d_input_std_deviation;
// ... (分配设备内存的代码省略)

// 5. 拷贝输入数据到设备
hcMemcpyAsync(d_input, h_input, input_size, hcMemcpyHostToDevice, stream);
hcMemcpyAsync(d_weight, h_weight, weight_size, hcMemcpyHostToDevice, stream);
if (bias_desc) {
    hcMemcpyAsync(d_bias, h_bias, bias_size, hcMemcpyHostToDevice, stream);
}

// 6. 执行Layer Normalization
status = layer_norm_desc->calculate(
    d_workspace,
    workspace_size,
    d_output,
    d_input_standardization,
    d_input_std_deviation,
    d_input,
    d_weight,
    d_bias,
    stream
);

// 7. 拷贝结果回主机
hcMemcpyAsync(h_output, d_output, output_size, hcMemcpyDeviceToHost, stream);
hcMemcpyAsync(h_std, d_input_std_deviation, std_size, hcMemcpyDeviceToHost, stream);
hcStreamSynchronize(stream);

// 8. 清理资源
delete layer_norm_desc;
hcFree(d_workspace);
// ... (释放其他资源的代码省略)
```

## 5. 实现细节

### 内存管理
- **Workspace策略**: 在设备端分配连续内存存储4个步长数组（input_strides, output_strides, input_standardization_strides, input_std_deviation_strides），每个数组大小为`ndim * sizeof(ptrdiff_t)`，总计`4 * ndim * sizeof(ptrdiff_t)`字节。主机端步长通过`hcMemcpyAsync`异步拷贝到设备，避免在kernel中重复计算
- **资源生命周期**: Descriptor持有Opaque结构体指针，Opaque内部通过`std::shared_ptr`管理`device::metax::Handle::Internal`，确保设备句柄的底层资源在使用期间不会被释放

### 并发性
- **异步执行**: 使用`hcMemcpyAsync`进行主机到设备的步长拷贝，kernel启动与stream中的其他操作异步执行，支持stream并发
- **线程安全**: Descriptor创建后只读，多个stream可并发使用同一Descriptor调用`calculate()`，但每个stream需要独立的workspace

### 性能
- **Kernel启动配置**: 使用2D网格`dim3(info.input_shape[0], info.input_shape[1])`，每个block对应一个归一化实例（如batch中的一个序列），block内仅使用1个线程（`BLOCK_SIZE=1`），通过复用CUDA实现的`layerNormKernel`完成计算
- **设备适配**: 根据设备的`maxThreadsPerBlock`属性（512或1024）选择kernel模板参数，虽然当前实现固定使用`BLOCK_SIZE=1`，但为未来优化预留了灵活性
- **计算复杂度**: 每个归一化实例的复杂度为O(normalized_size)，包括两次遍历（计算均值和方差）和一次遍历（应用仿射变换），使用CUB的`BlockReduce`进行高效的block内归约

### 错误处理
- **数据类型验证**: 仅支持`INFINI_DTYPE_F16`、`INFINI_DTYPE_F32`、`INFINI_DTYPE_BF16`，其他类型返回`INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状验证**: 通过`LayerNormInfo::createLayerNormInfo`验证输入、输出、权重、偏置张量的形状兼容性，不匹配时返回`INFINI_STATUS_BAD_TENSOR_SHAPE`
- **Workspace检查**: `calculate()`方法验证workspace_size >= _workspace_size，不足时返回`INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **设备支持检查**: 如果设备的maxThreadsPerBlock既不是512也不是1024，返回`INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
- **API错误处理**: 使用`CHECK_METAX`宏检查Metax API调用（如`hcMemcpyAsync`），失败时返回相应错误码

### 依赖关系
- **外部依赖**:
  - Metax SDK: `<mcblas/mcblas.h>`, `<mcdnn/mcdnn.h>` (ENABLE_METAX_MC_API定义时) 或 `<hcblas/hcblas.h>`, `<hcdnn/hcdnn.h>` (默认)
  - CUB或兼容库: `<mccub/block/block_reduce.cuh>` 或 `<hccub/block/block_reduce.cuh>`
- **内部依赖**:
  - `../layer_norm.h`: 提供DESCRIPTOR宏定义，生成标准接口
  - `../cuda/kernel.cuh`: 提供`layerNormKernel`设备函数实现（通过模板复用）
  - `../info.h`: 提供`LayerNormInfo`类用于形状验证和元数据存储
  - `../../../devices/metax/metax_common.h`: 提供Metax设备句柄和资源池管理
  - `../../../devices/metax/metax_kernel_common.h`: 提供kernel宏定义和工具函数（如`INFINIOP_METAX_KERNEL`、`METAX_BLOCK_SIZE_1024`）
  - `../../../reduce/cuda/reduce.cuh`: 提供`sum`和`sumSquared`归约原语

### 设计模式
- **桥接模式 (Bridge Pattern)**: 通过`Opaque`结构体将设备特定的实现（`device::metax::Handle::Internal`）与通用接口分离，支持不同Metax设备架构
- **模板方法模式 (Template Method)**: `DESCRIPTOR`宏定义了Descriptor类的结构模板，各后端（cuda、metax、cpu等）实现create和calculate方法
- **策略模式 (Strategy Pattern)**: 根据设备的maxThreadsPerBlock属性选择不同的BLOCK_SIZE模板参数（虽然当前实现均使用1，但为多架构支持预留了策略切换逻辑）
- **RAII (Resource Acquisition Is Initialization)**: Descriptor构造时初始化资源，析构时自动释放Opaque，避免资源泄漏
