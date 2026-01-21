# Softplus Metax 操作符核心实现文档

本文档详细描述了 Softplus 激活函数在 Metax 设备上的实现。该模块基于 InfiniOP 的 elementwise 基础设施，为 Metax GPU（沐曦加速卡）提供高性能的 Softplus 操作实现。

## 1. 模块结构

- **`softplus_metax.h`**: Metax 后端的 API 声明，通过 ELEMENTWISE_DESCRIPTOR 宏定义描述符类
- **`softplus_metax.maca`**: Metax 后端的具体实现，包含描述符创建和计算的核心逻辑

## 2. 核心类

### `Descriptor`
- **位置**: `softplus_metax.h`（通过宏展开定义）
- **主要功能**: 封装 Softplus 操作在 Metax 设备上的执行描述符，管理设备实现、数据类型、张量形状和 workspace 大小
- **关键成员**:
  - `_dtype: infiniDtype_t`: 操作的数据类型（支持 F16, BF16, F32, F64）
  - `_info: op::elementwise::ElementwiseInfo`: 张量形状、步幅、广播和连续性信息
  - `_device_info: std::unique_ptr<op::elementwise::metax::DeviceImpl>`: Metax 设备特定的实现对象，负责实际的 kernel 启动
  - `_workspace_size: size_t`: 所需工作空间大小，用于存储元数据和输入指针数组
- **核心方法**:
  - `~Descriptor()`: 默认析构函数
  - `create(handle, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，创建并初始化描述符
    - 验证数据类型（F16, BF16, F32, F64）
    - 检查输入输出张量形状一致性
    - 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏初始化底层实现
    - 返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 Softplus 计算
    - 检查 workspace 大小是否足够
    - 根据 `_dtype` 分发到对应的模板实例化
    - 调用 `_device_info->calculate<256, cuda::SoftplusOp, T>()` 启动 Metax kernel
    - 使用 256 线程/块的块大小
- **生命周期**: 由用户通过 `infiniopCreateSoftplusDescriptor` 创建，通过 `infiniopDestroySoftplusDescriptor` 销毁

## 3. API 接口

```cpp
// 创建 Softplus 描述符（通过 operator.cc 调用）
infiniStatus_t op::softplus::metax::Descriptor::create(
    infiniopHandle_t handle_,                    // Metax 设备句柄
    Descriptor **desc_ptr,                       // 输出：创建的描述符指针
    infiniopTensorDescriptor_t out_desc,         // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量
);
// 验证数据类型和形状，初始化 ElementwiseInfo 和 DeviceImpl

// 执行 Softplus 计算
infiniStatus_t op::softplus::metax::Descriptor::calculate(
    void *workspace,                             // 设备内存 workspace
    size_t workspace_size,                       // workspace 大小
    void *output,                                // 输出张量设备指针
    std::vector<const void *> inputs,            // 输入张量设备指针向量
    void *stream                                 // hcStream_t 流
) const;
// 启动 Metax kernel 执行 log1p(exp(x)) 操作，对于大值（x>20）直接返回 x
```

## 4. 使用示例

```cpp
// 示例：在 Metax 设备上执行 Softplus 激活函数
#include "infiniop/ops/softplus.h"
#include "metax/softplus_metax.h"

// 1. 创建 Metax 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, device_id);

// 2. 创建输入输出张量描述符
int64_t shape[] = {1024, 1024};
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F32, 2, shape, nullptr);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F32, 2, shape, nullptr);

// 3. 创建 Softplus 描述符
infiniopSoftplusDescriptor_t softplus_desc;
infiniopCreateSoftplusDescriptor(handle, &softplus_desc, y_desc, x_desc);

// 4. 分配 workspace
size_t workspace_size;
infiniopGetSoftplusWorkspaceSize(softplus_desc, &workspace_size);
void *workspace;
hcMalloc(&workspace, workspace_size);

// 5. 分配输入输出设备内存
float *d_x, *d_y;
hcMalloc((void**)&d_x, 1024 * 1024 * sizeof(float));
hcMalloc((void**)&d_y, 1024 * 1024 * sizeof(float));

// 6. 拷贝输入数据到设备
hcMemcpyAsync(d_x, h_x, 1024 * 1024 * sizeof(float), hcMemcpyHostToDevice, stream);

// 7. 执行 Softplus 计算
infiniopSoftplus(softplus_desc, workspace, workspace_size, d_y, d_x, stream);

// 8. 拷贝结果回主机
hcMemcpyAsync(h_y, d_y, 1024 * 1024 * sizeof(float), hcMemcpyDeviceToHost, stream);

// 9. 清理资源
hcFree(d_x);
hcFree(d_y);
hcFree(workspace);
infiniopDestroySoftplusDescriptor(softplus_desc);
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 数学定义
Softplus 激活函数的数学定义为：
```
softplus(x) = log(1 + exp(x)) = log1p(exp(x))
```

对于数值稳定性，当 x > 20 时，直接返回 x（因为 exp(20) >> 1，此时 log1p(exp(x)) ≈ x）。

### 内存管理
- **Workspace 策略**: workspace 用于存储以下元数据：
  - 输入指针数组（`N * sizeof(void*)`，N 为输入数量）
  - 形状和步 strides 数据（输出形状、输出步幅、所有输入形状、所有输入步幅）
  - 连续性和广播标志
- **数据传输**: 在每次计算前，通过 `hcMemcpyAsync` 将元数据从主机复制到设备 workspace
- **零拷贝**: 输入输出张量数据直接在设备内存上操作，无需额外拷贝

### 并发
- **Kernel 启动**: 使用 CUDA-style 的 `<<<grid, block, 0, stream>>>` 语法启动 Metax kernel
- **线程配置**: 固定使用 256 线程/块（`BLOCK_SIZE = 256`）
- **Grid 大小**: 根据输出大小动态计算，受限于设备的 `maxThreadsPerBlock` 和 `gridSizeX`
- **流支持**: 完全支持异步执行，接受 `hcStream_t` 流参数

### 性能
- **Block 大小**: 使用 256 线程/块的启发式配置，平衡了寄存器使用和 occupancy
- **循环启动**: 对于大型张量，使用循环分段执行（每次处理 `gridDims.x * blockDims.x` 个元素）
- **分支优化**: CUDA kernel 中的 SoftplusOp 使用 `if constexpr` 编译期分支，避免运行时开销
- **向量化**: 对 half2 类型提供了向量化实现（虽然 Softplus 标量实现未使用）
- **复杂度**: O(n) 时间复杂度，n 为输出张量的元素数量

### 错误处理
- **错误码传播**: 使用 `INFINI_STATUS_*` 枚举返回详细错误状态
- **数据类型验证**: 只支持 F16, BF16, F32, F64，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状检查**: 通过 `CHECK_SAME_SHAPE` 宏确保输入输出形状一致
- **Workspace 验证**: 计算前检查 workspace 大小，不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **Result 类型**: ElementwiseInfo 创建使用 `utils::Result<T>` 模式处理可能的错误

### 依赖
- **外部依赖**:
  - `infiniop/ops/elementwise/metax/elementwise_metax_api.h`: 提供 ElementwiseInfo 和 DeviceImpl 基础设施
  - `infiniop/ops/elementwise/metax/elementwise_metax.h`: 提供 Metax kernel 启动逻辑
  - `infiniop/ops/softplus/cuda/kernel.cuh`: 提供 `cuda::SoftplusOp` 算子实现（Metax 复用 CUDA kernel）
  - Metax 驱动运行时（`hcMemcpyAsync`, `hcStream_t`, `hcMalloc`）
- **宏依赖**:
  - `ELEMENTWISE_DESCRIPTOR`: 自动生成描述符类结构
  - `CREATE_ELEMENTWISE_METAX_DESCRIPTOR`: 自动初始化 MetAX 特定的描述符字段
  - `CHECK_DTYPE`: 数据类型验证宏
  - `CHECK_SAME_SHAPE`: 形状一致性检查宏

### 设计模式
- **模板方法模式**: `DeviceImpl::calculate` 使用模板参数分发到不同的数据类型实现
- **策略模式**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏，Softplus 复用整个 elementwise 基础设施
- **RAII**: 使用 `std::unique_ptr` 管理 `DeviceImpl`，自动释放资源
- **类型擦除**: `Descriptor` 继承自 `InfiniopDescriptor`，通过基类指针实现多态
- **编译期多态**: 使用模板和 `if constexpr` 实现零开销抽象

### Kernel 实现（CUDA，Metax 复用）
Softplus 的核心计算逻辑在 `cuda/kernel.cuh` 中定义：

```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &x) const {
    if constexpr (std::is_same_v<T, half>) {
        // FP16: 提升到 float 计算以保证数值稳定性
        float xf = __half2float(x);
        float out = (xf > 20.0f) ? xf : log1pf(expf(xf));
        return __float2half(out);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        // BF16: 同样提升到 float
        float xf = __bfloat162float(x);
        float out = (xf > 20.0f) ? xf : log1pf(expf(xf));
        return __float2bfloat16(out);
    } else {
        // FP32/FP64: 直接计算
        return (x > T(20)) ? x : log1p(exp(x));
    }
}
```

**数值稳定性考虑**:
- 对于半精度类型（FP16, BF16），先提升到 FP32 计算，避免溢出和精度损失
- 对于 x > 20 的情况，直接返回 x，避免计算 `exp(20)` 导致的溢出
- 使用 `log1p` 而非 `log(1 + x)`，在小 x 时保持精度

**Metax 兼容性**:
- Metax backend 直接复用 CUDA kernel 实现文件（`#include "../cuda/kernel.cuh"`）
- Metax 设备通过沐曦的编译工具链支持 CUDA-style 语法，因此无需重写 kernel
- 仅需在 host 端（`.maca` 文件）调整设备初始化和 kernel 启动逻辑
