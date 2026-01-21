# Metax 张量减法运算 (Subtraction Operation) 核心实现文档

本模块实现了在 Moore Threads Metax GPU 设备上的张量逐元素减法运算。通过继承统一的逐元素运算框架，为 Metax 硬件后端提供高效的减法计算能力，支持 FP16、FP32、FP64 和 BF16 四种浮点数据类型。

## 1. 模块结构

- **`sub_metax.h`**: 公共 API 定义，使用 `ELEMENTWISE_DESCRIPTOR` 宏声明 `Descriptor` 类，将减法运算注册到逐元素运算框架
- **`sub_metax.maca`**: 核心实现文件，包含描述符的创建与计算调度逻辑，复用 CUDA 内核函数实现具体运算

## 2. 核心类

### `op::sub::metax::Descriptor`
- **位置**: `sub_metax.maca` (通过 `ELEMENTWISE_DESCRIPTOR` 宏生成)
- **主要功能**: Metax 设备上的张量减法运算描述符，管理运算元数据、工作空间需求及设备实现
- **关键成员**:
  - `_dtype: infiniDtype_t`: 运算数据类型 (F16/F32/F64/BF16)
  - `_info: op::elementwise::ElementwiseInfo`: 张量形状、步长、连续性等元数据
  - `_device_info: std::unique_ptr<op::elementwise::metax::DeviceImpl>`: Metax 设备具体实现指针
  - `_workspace_size: size_t`: 设备端所需工作空间大小 (存储输入指针数组和元数据)
- **核心方法**:
  - `create(infiniopHandle_t handle_, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, std::vector<infiniopTensorDescriptor_t> input_desc_vec)`:
    - **功能**: 创建减法运算描述符，验证输入输出张量兼容性并初始化设备实现
    - **算法**:
      1. 提取 Metax 设备句柄和数据类型
      2. 验证数据类型必须是浮点类型 (F16/F32/F64/BF16)
      3. 检查输入输出张量形状一致性 (通过 `CHECK_SAME_SHAPE` 宏)
      4. 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏，构建 `ElementwiseInfo` 元数据并创建 `DeviceImpl` 实例
    - **复杂度**: O(n)，n 为张量维度数
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const`:
    - **功能**: 在 Metax GPU 上执行减法计算
    - **算法**:
      1. 验证工作空间大小是否满足需求
      2. 根据数据类型分发到对应的模板实例化，调用 `DeviceImpl::calculate<256, cuda::SubOp, T>()`
      3. 使用 256 线程块大小和 CUDA 定义的 `SubOp` 函数对象
    - **复杂度**: O(m)，m 为输出张量元素总数
- **生命周期**: 通过 `create` 静态方法构造，析构函数默认实现释放资源

## 3. API 接口

```cpp
// 元运算符定义 (复用 CUDA 实现)
namespace op::sub::cuda {
    struct SubOp {
        static constexpr size_t num_inputs = 2;

        template <typename T>
        __device__ __forceinline__ T operator()(const T &a, const T &b) const;

        // 对 half2/float2 类型使用向量指令 __hsub2
        // 对 half/float 使用标量指令 __hsub/__fsub_rd
        // 对其他类型使用二元运算符 -
    };
}

// 描述符创建接口
infiniStatus_t op::sub::metax::Descriptor::create(
    infiniopHandle_t handle_,              // Metax 设备句柄
    Descriptor **desc_ptr,                 // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t out_desc,   // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // {A, B} 输入张量描述符
);
// 返回 INFINI_STATUS_SUCCESS 或错误码

// 计算执行接口
infiniStatus_t op::sub::metax::Descriptor::calculate(
    void *workspace,                       // 设备端工作空间 (存储元数据)
    size_t workspace_size,                 // 工作空间大小
    void *output,                          // 输出张量设备指针
    std::vector<const void *> inputs,      // {A_ptr, B_ptr} 输入张量设备指针
    void *stream                           // Metax 计算流 (hcStream_t)
) const;
// 返回 INFINI_STATUS_SUCCESS 或 INFINI_STATUS_INSUFFICIENT_WORKSPACE
```

## 4. 使用示例

```cpp
// 示例：在 Metax GPU 上执行张量减法 C = A - B

// 1. 准备张量描述符 (假设形状为 {1024, 1024})
infiniopTensorDescriptor_t desc_A, desc_B, desc_C;
infiniopCreateTensorDescriptor(&desc_A, INFINI_DTYPE_F32, {1024, 1024}, nullptr);
infiniopCreateTensorDescriptor(&desc_B, INFINI_DTYPE_F32, {1024, 1024}, nullptr);
infiniopCreateTensorDescriptor(&desc_C, INFINI_DTYPE_F32, {1024, 1024}, nullptr);

// 2. 创建 Metax 设备句柄和减法描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);

op::sub::metax::Descriptor *sub_desc = nullptr;
auto status = op::sub::metax::Descriptor::create(
    handle, &sub_desc, desc_C, {desc_A, desc_B}
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 分配设备内存并初始化输入数据
void *d_A, *d_B, *d_C;
size_t tensor_size = 1024 * 1024 * sizeof(float);
hcMalloc(&d_A, tensor_size);
hcMalloc(&d_B, tensor_size);
hcMalloc(&d_C, tensor_size);

// 从主机复制数据到设备 (省略数据初始化)
hcMemcpyAsync(d_A, h_A, tensor_size, hcMemcpyHostToDevice, stream);
hcMemcpyAsync(d_B, h_B, tensor_size, hcMemcpyHostToDevice, stream);

// 4. 分配工作空间并执行减法运算
size_t workspace_size = sub_desc->workspaceSize();
void *d_workspace;
hcMalloc(&d_workspace, workspace_size);

status = sub_desc->calculate(
    d_workspace, workspace_size,
    d_C, {d_A, d_B},
    stream
);

// 5. 同步并取回结果
hcStreamSynchronize(stream);
hcMemcpyAsync(h_C, d_C, tensor_size, hcMemcpyDeviceToHost, stream);

// 6. 清理资源
delete sub_desc;
hcFree(d_A); hcFree(d_B); hcFree(d_C); hcFree(d_workspace);
```

## 5. 实现细节

### 内存管理
- **工作空间布局**: 采用两级结构，前段存储输入指针数组 (N * sizeof(void*))，后段存储 `ElementwiseInfo` 元数据
  - 元数据包含: 输出形状/步长、所有输入的形状/步长、输入连续性标志、广播标志
  - 通过 `infoToDevice` 模板函数使用 `hcMemcpyAsync` 异步拷贝到设备
- **设备端内存布局**: 使用指针偏移计算从单一工作空间基址提取各元数据段，避免多次分配

### 并发性
- **执行模型**: 基于 Metax HIP 内核的 SIMD 并行，使用 256 线程/块
- **网格-块配置**:
  - 块大小: `min(256, maxThreadsPerBlock)` (适配设备限制)
  - 网格大小: `min(ceil(output_size / block_size), gridSizeX)` (防止超过网格 X 维度上限)
  - 分步执行: 对于超大型张量，通过循环分段启动内核 (`for (size_t i = 0; i < output_size; i += step)`)
- **线程安全**: 描述符创建阶段为只读操作，计算阶段通过独立的 stream 隔离

### 性能优化
- **向量化指令**: 对 FP16/FP32 的双精度向量类型 (half2, cuda_bfloat162) 使用 `__hsub2` SIMD 指令，单指令处理两个元素
- **连续内存路径**: 通过 `isOutputContiguous` 和 `input_contiguous[]` 标志识别连续张量，使用线性索引避免 `indexToOffset` 坐标转换开销
- **广播支持**: `InputIndexer` 结构体封装广播逻辑，自动处理形状不匹配的输入张量 (如 (3,1) 广播到 (3,4))
- **模板特化**: 编译期为每种数据类型生成专用内核实例，避免运行时分支

### 错误处理
- **验证机制**:
  - 数据类型检查: `CHECK_DTYPE` 宏限制为四种浮点类型
  - 形状兼容性: `CHECK_SAME_SHAPE` 宏确保所有输入输出张量形状一致
  - 工作空间大小: 运行时检查 `workspace_size < _workspace_size` 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **错误传播**: 使用 `CHECK_RESULT` 和 `CHECK_STATUS` 宏将底层错误码向上传播
- **降级处理**: 当输出张量为空 (size=0) 时直接返回成功，避免无效内核启动

### 依赖关系
- **外部依赖**:
  - `elementwise_metax_api.h`: 提供 `DeviceImpl` 类和 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏
  - `sub/cuda/kernel.cuh`: 提供 `SubOp` 函数对象 (复用 CUDA 实现，因为 Metax 兼容 CUDA 编程模型)
  - Metax HIP Runtime (`hcMemcpyAsync`, `hcMalloc`, `<<<grid, block, 0, stream>>>` 内核启动语法)
- **内部依赖**:
  - `ElementwiseInfo`: 封装张量元数据管理
  - `device::metax::Handle::Internal`: 设备句柄内部表示，提供 `maxThreadsPerBlock()` 和 `gridSizeX()` 查询

### 设计模式
- **模板方法模式**: `ELEMENTWISE_DESCRIPTOR` 宏生成标准化的描述符类模板，子类 (如 sub) 仅实现 create/calculate 逻辑
- **策略模式**: `DeviceImpl` 封装设备特定的内核启动策略，通过模板参数 `Op` (SubOp) 和数据类型组合实现多态
- **RAII**: 使用 `std::unique_ptr` 管理 `DeviceImpl` 生命周期，确保资源释放
- **编译期多态**: 基于 C++ 模板和 `constexpr` 成员变量 (`num_inputs`) 在编译期完成操作符分发
