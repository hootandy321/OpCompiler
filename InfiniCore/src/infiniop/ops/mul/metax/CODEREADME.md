# Metax 逐元素乘法 (Element-wise Multiplication) 核心实现文档

本模块实现了基于沐曦 Metax GPU 设备的逐元素张量乘法运算 (`output = input1 * input2`)，支持逐元素广播机制和多种浮点数据类型（FP16、FP32、FP64、BF16）。作为逐元素运算的特化实例，该模块复用通用的 elementwise_metax 框架，仅通过 CUDA 算子 functor 定义乘法语义。

## 1. 模块结构

- **`mul_metax.h`**: 头文件，通过 `ELEMENTWISE_DESCRIPTOR` 宏生成 `op::mul::metax::Descriptor` 类声明，导出创建和计算接口
- **`mul_metax.maca`**: 实现文件，定义 Descriptor 类的析构函数、`create` 工厂方法和 `calculate` 执行方法，处理类型分发和 Metax 设备启动

## 2. 核心类

### `op::mul::metax::Descriptor`
- **位置**: `mul_metax.h`（宏生成）、`mul_metax.maca`（方法实现）
- **主要功能**: Metax 设备上逐元素乘法运算的算子描述符，管理运算元数据、设备实现实例和工作空间需求
- **关键成员**:
  - `infiniDtype_t _dtype`: 目标数据类型（FP16/FP32/FP64/BF16）
  - `op::elementwise::ElementwiseInfo _info`: 封装张量形状、步长、连续性、广播信息等元数据
  - `std::unique_ptr<op::elementwise::metax::DeviceImpl> _device_info`: Metax 设备端实现，负责 kernel 启动和内存管理
  - `size_t _workspace_size`: 所需工作空间大小（存储输入指针数组和元数据拷贝）
- **核心方法**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，验证张量描述符一致性，构建 `ElementwiseInfo`，创建 `DeviceImpl`，计算工作空间大小并实例化 Descriptor。使用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏封装流程，通过 `CHECK_SAME_SHAPE` 确保输出和所有输入形状完全一致（逐元素运算不要求自动广播，需用户手动对齐）
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行乘法计算。首先验证工作空间大小，然后根据 `_dtype` 分发到对应的模板实例化，调用 `_device_info->calculate<256, cuda::MulOp, T>()` 启动 Metax kernel。模板参数 `256` 为 CUDA block 大小，`cuda::MulOp` 为乘法 functor
- **生命周期**: 由 `create` 静态方法构造，通过 `std::unique_ptr` 管理 `DeviceImpl`，析构时自动释放资源

## 3. API 接口

```cpp
namespace op::mul::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // 获取所需工作空间大小
    size_t workspaceSize() const;

    // 创建描述符实例
    static infiniStatus_t create(
        infiniopHandle_t handle_,                          // Metax 设备句柄
        Descriptor **desc_ptr,                             // 输出：描述符指针
        infiniopTensorDescriptor_t out_desc,               // 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符 [input1, input2]
    );

    // 执行逐元素乘法
    infiniStatus_t calculate(
        void *workspace,               // 设备工作空间指针
        size_t workspace_size,         // 工作空间大小（字节）
        void *output,                  // 输出张量设备指针
        std::vector<const void *> inputs, // 输入张量设备指针 [input1_ptr, input2_ptr]
        void *stream                   // Metax 计算流 (hcStream_t)
    ) const;
};

} // namespace op::mul::metax
```

## 4. 使用示例

```cpp
// 示例：在 Metax GPU 上执行逐元素张量乘法
// 假设已初始化 handle、创建张量描述符并分配设备内存

infiniopHandle_t handle;              // Metax 设备句柄
infiniopTensorDescriptor_t x_desc, y_desc, z_desc;  // 输入和输出描述符
half *d_x, *d_y, *d_z;                // 设备内存指针（FP16 类型）
hcStream_t stream;                    // Metax 计算流

// 1. 创建乘法算子描述符
op::mul::metax::Descriptor *mul_desc = nullptr;
infiniStatus_t status = op::mul::metax::Descriptor::create(
    handle,
    &mul_desc,
    z_desc,          // 输出描述符
    {x_desc, y_desc} // 输入描述符数组
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（形状不匹配、不支持的数据类型等）
}

// 2. 分配工作空间
size_t workspace_size = mul_desc->workspaceSize();
void *d_workspace = nullptr;
hcMalloc(&d_workspace, workspace_size);

// 3. 执行乘法运算：z = x * y
status = mul_desc->calculate(
    d_workspace,
    workspace_size,
    d_z,             // 输出：d_z[i] = d_x[i] * d_y[i]
    {d_x, d_y},      // 输入指针数组
    stream
);

// 4. 同步并等待计算完成
hcStreamSynchronize(stream);

// 5. 清理资源
hcFree(d_workspace);
delete mul_desc;
```

## 5. 实现细节

- **内存管理**:
  - 工作空间布局：首先存储输入指针数组（`N * sizeof(void*)`，N=2），随后拷贝 `ElementwiseInfo` 的元数据块（形状、步长、连续性标志、广播标志）
  - 元数据通过 `hcMemcpyAsync` 异步拷贝到设备，避免 host-device 同步开销
  - 使用 `std::unique_ptr` 管理 `DeviceImpl`，确保异常安全和自动释放

- **并发性**:
  - 运算在 Metax stream 上异步执行，支持多个算子并行调度到不同流
  - kernel 启动采用网格步进策略（grid-stride loop），通过 `step = gridDims.x * blockDims.x` 和循环 `for (size_t i = 0; i < output_size; i += step)` 处理超大规模张量，避免单次 kernel 启动的线程数限制
  - 设备侧无显式锁机制，依赖数据并行模型（每个线程处理独立元素）

- **性能优化**:
  - **Block 大小**: 固定为 256 线程，平衡 occupancy 和寄存器压力
  - **向量化指令**: `cuda::MulOp` functor 针对不同数据类型使用专用 intrinsic
    - `half2` / `cuda_bfloat162`: 调用 `__hmul2` 进行 2 元素 SIMD 乘法
    - `half` / `cuda_bfloat16`: 调用 `__hmul` 标量乘法
    - `float`: 调用 `__fmul_rn` IEEE 754 舍入乘法
    - `double`: 直接使用 `*` 运算符（编译器自动向量化）
  - **连续性优化**: `ElementwiseInfo` 记录每个张量的连续性，连续张量直接使用线性索引 `idx`，非连续张量调用 `device::metax::indexToOffset` 计算偏移量，减少地址计算开销
  - **网格大小**: 动态计算为 `std::min(ceil_div(output_size, BLOCK_SIZE), max_grid_size)`，充分利用设备并行能力

- **错误处理**:
  - `create` 方法使用 `CHECK_DTYPE` 验证数据类型（仅支持 FP16/FP32/FP64/BF16），否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 使用 `CHECK_SAME_SHAPE` 宏验证输出和所有输入形状完全一致，否则返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
  - `calculate` 方法检查工作空间大小，不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
  - 不支持的 dtype 分发到 default 分支，返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 底层 Metax 调用（`hcMemcpyAsync`, `kernel<<<>>>`）通过 `CHECK_METAX` 宏捕获错误并转换为 `infiniStatus_t`

- **依赖关系**:
  - **外部依赖**:沐曦 Metax Runtime (`hcMemcpyAsync`, `hcMalloc`, `hcStream_t`), CUDA intrinsic 头文件（`__hmul2`, `__hmul`, `__fmul_rn`）
  - **模块间依赖**:
    - `op::elementwise::metax::DeviceImpl`: 提供 `calculate` 模板方法，封装 kernel 启动逻辑
    - `op::elementwise::ElementwiseInfo`: 管理张量元数据和布局信息
    - `op::mul::cuda::MulOp`: 定义乘法语义的 CUDA functor，支持 `operator()(T a, T b)` 重载
    - `device::metax::Handle`: Metax 设备句柄，提供设备属性访问（maxThreadsPerBlock, gridSizeX）

- **设计模式**:
  - **Template Method**: `ELEMENTWISE_DESCRIPTOR` 宏生成标准化的 Descriptor 结构，子类通过实现 `create` 和 `calcualte` 定制行为
  - **Strategy Pattern**: `cuda::MulOp` functor 封装乘法算法，可通过模板参数替换为其他逐元素运算（如加、减、除）
  - **Factory Pattern**: `create` 静态方法作为工厂，封装描述符构建流程
  - **Bridge Pattern**: `Descriptor`（host 侧接口）与 `DeviceImpl::Opaque`（device 侧实现）分离，支持多后端扩展
