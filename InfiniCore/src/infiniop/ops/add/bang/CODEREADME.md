# Add Operation for BANG (Cambricon MLU) Implementation

本模块实现了在寒武纪（Cambricon）MLU硬件上的张量加法操作，采用高度优化的逐元素（elementwise）操作框架。该实现通过BANG编程模型利用MLU的NRAM（近端内存）进行高效计算，支持F16、BF16和F32三种浮点数据类型，并自动处理广播、非连续内存布局等复杂场景。

## 1. Module Structure

- **`add_bang.h`**: 头文件，通过ELEMENTWISE_DESCRIPTOR宏声明加法操作的BANG后端接口，定义Descriptor类的基本结构。
- **`add_bang.mlu`**: 主实现文件，包含Descriptor类的完整实现，负责操作符创建、类型分发和计算调度。
- **`add_bang_internal.mlu`**: 内核实现文件，定义AddOp仿函数并实例化模板化的MLU设备内核。

## 2. Core Classes

### `AddOp` (Internal Functor)
- **Location**: `add_bang_internal.mlu:6-17`
- **Primary Function**: 定义加法操作的设备端计算逻辑，作为可调用对象传递给通用逐元素操作框架
- **Key Members**:
  - `num_inputs`: 静态常量，值为2，指定此操作符接受2个输入张量
- **Core Methods**:
  - `operator()(T *out, const T *a, const T *b, size_t num_elements)`: 执行设备端加法运算
    - 对于`half`、`bfloat16_t`、`float`类型，调用BANG库函数`__bang_add`进行硬件加速
    - 对于其他类型，使用简单的逐元素加法（fallback）
    - 该函数标记为`__mlu_device__`，仅在MLU设备上执行
- **Lifecycle**: 编译期静态结构，无运行时构造/析构开销

### `Descriptor` (Public API)
- **Location**: `add_bang.mlu:17-42` (定义于ELEMENTWISE_DESCRIPTOR宏展开)
- **Primary Function**: 加法操作的描述符类，继承自InfiniopDescriptor，管理操作的元数据和设备实现
- **Key Members**:
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型（F16/BF16/F32）
  - `_info`: `op::elementwise::ElementwiseInfo`，封装输入/输出张量的形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::bang::DeviceImpl>`，BANG设备实现的句柄
  - `_workspace_size`: `size_t`，设备端工作空间所需的字节数
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，构造加法操作描述符
    - 验证数据类型必须是F16、BF16或F32
    - 检查输入/输出张量形状完全一致（不支持广播）
    - 调用`CREATE_ELEMENTWISE_BANG_DESCRIPTOR`宏创建ElementwiseInfo和DeviceImpl
    - 返回`INFINI_STATUS_SUCCESS`或错误码
  - `calculate(workspace, workspace_size, output, inputs, queue)`: 执行加法计算
    - 验证工作空间大小是否足够
    - 根据`_dtype`分发到对应类型的模板特化：`half`、`bfloat16_t`或`float`
    - 调用`_device_info->calculate<AddOp, Tdata>`启动设备内核
  - `workspaceSize()`: 返回所需工作空间大小
  - `~Descriptor()`: 默认析构函数
- **Lifecycle**: 由用户通过`create`方法构造，使用完毕后由用户负责释放

### `op::elementwise::bang::DeviceImpl` (Infrastructure)
- **Location**: `elementwise_bang.h:15-144` (在elementwise模块中定义)
- **Primary Function**: BANG设备逐元素操作的底层实现，管理元数据传输和内核启动
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>`，指向内部实现的Pimpl对象
  - `Opaque::internal`: `std::shared_ptr<device::bang::Handle::Internal>`，MLU设备句柄的内部状态
- **Core Methods**:
  - `calculateImpl<N, Op, Tdata>(info, workspace, output, inputs, queue, args...)`: 核心计算逻辑
    - 将主机端的元数据（形状、步长、连续性标志）拷贝到设备端工作空间
    - 拷贝输入张量指针数组到设备
    - 调用`Op::launch`启动MLU内核（对于Add操作即`launchAddKernel`）
    - 同步队列（`cnrtQueueSync`）确保计算完成
  - `infoToDevice<N>(...)`: 辅助函数，将ElementwiseInfo的元数据打包到设备内存
    - 在工作空间中依次布局：输入指针数组、输出形状、输出步长、所有输入形状、所有输入步长、连续性标志、广播标志
    - 使用`cnrtMemcpy`进行主机到设备的异步拷贝
- **Lifecycle**: 通过`DeviceImpl::create(handle->internal())`构造，由Descriptor独占管理

### `elementwiseKernel` (MLU Kernel)
- **Location**: `elementwise_bang_kernel.mlu:153-213`
- **Primary Function**: 在MLU设备上执行的通用逐元素操作内核，由多个MLU核心并行执行
- **Key Members**:
  - `typed_inputs[N]`: 类型化的输入指针数组（从void*转换而来）
  - `nram_buf`: `__nram__ Tdata[NRAM_MAX_SIZE/sizeof(Tdata)]`，NRAM上的缓冲区，用于暂存输入和输出数据
  - `input_indexes[N]`: 每个输入张量的起始索引偏移
- **Core Methods**:
  - `__mlu_global__ void elementwiseKernel(...)`: 内核入口函数
    - 计算当前任务（task）负责处理的数据范围：`[start_idx, end_idx)`
    - 分配NRAM缓冲区（使用`__nram__`关键字）
    - 调用`getOutputIndex`和`InputIndexer`计算输入/输出的内存偏移
    - 调用`launchOp`执行实际计算
  - `launchOp<N, Op, Tdata>(...)`: 设备端计算核心
    - 计算NRAM可用空间：`nram_usable = NRAM_MAX_SIZE - ALIGN_SIZE*(N+1)`
    - 计算最大批次大小：`max_batch = nram_usable / ((N+1)*sizeof(Tdata))`
    - 循环处理数据，每轮处理`curr_batch`个元素：
      1. 对于连续输入，使用`__memcpy_async(GDRAM2NRAM)`批量拷贝
      2. 对于非连续输入，调用`nonContiguousMemcpy`进行分段拷贝
      3. 调用`Op::operator()`（即AddOp）在NRAM中执行计算
      4. 使用`__memcpy_async(NRAM2GDRAM)`将结果写回全局内存
    - 使用`__sync_io()`和`__sync_compute()`同步内存和计算操作
- **Lifecycle**: 作为CUDA类内核由`launchElementwiseKernelWrapper`启动

### `launchElementwiseKernelWrapper` (Host-side Launcher)
- **Location**: `elementwise_bang_kernel.mlu:223-264`
- **Primary Function**: 在主机端配置MLU内核启动参数并启动计算
- **Core Methods**:
  - 根据硬件信息获取每个集群的核心数和集群数量：`internal->getCorePerCluster()`, `getClusterCount()`
  - 设置内核启动维度：`dim.x = core_per_cluster`, `dim.y = cluster_count`
  - 根据问题规模选择内核类型：
    - 大规模连续操作（output_size > 1M）：使用`CNRT_FUNC_TYPE_UNION1`以获得更高性能
    - 其他情况：使用默认的`CNRT_FUNC_TYPE_BLOCK`
  - 使用`<<<dim, func_type, queue>>>`语法启动内核
- **Complexity**: O(n) 其中n为输出张量元素数量，在MLU核心间均匀分配

## 3. API Interface

```cpp
// 创建加法操作描述符
infiniStatus_t op::add::bang::Descriptor::create(
    infiniopHandle_t handle_,                    // [in] BANG设备句柄
    Descriptor **desc_ptr,                       // [out] 输出描述符指针
    infiniopTensorDescriptor_t out_desc,         // [in] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符数组（大小为2）
);
// 返回：INFINI_STATUS_SUCCESS，或数据类型/形状不匹配的错误码

// 执行加法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                             // [in] 设备工作空间指针
    size_t workspace_size,                       // [in] 工作空间大小（字节）
    void *output,                                // [out] 输出张量设备指针
    std::vector<const void *> inputs,            // [in] 输入张量设备指针数组（[a_ptr, b_ptr]）
    void *queue                                  // [in] BANG执行队列
) const;
// 返回：INFINI_STATUS_SUCCESS，或工作空间不足/数据类型错误的错误码

// 查询所需工作空间大小
size_t Descriptor::workspaceSize() const;
// 返回：计算所需的最小工作空间字节数

// 设备端加法仿函数（内核内部使用）
template <typename T>
struct AddOp {
    static constexpr size_t num_inputs = 2;

    __mlu_device__ void operator()(
        T *out,                                   // [out] 输出缓冲区（NRAM）
        const T *a,                               // [in] 第一个输入缓冲区（NRAM）
        const T *b,                               // [in] 第二个输入缓冲区（NRAM）
        size_t num_elements                       // [in] 待处理的元素数量
    ) const;
    // 对于T=half/bfloat16_t/float，调用__bang_add(out, a, b, num_elements)
    // 对于其他类型，执行逐元素加法循环
};
```

## 4. Usage Example

```cpp
// 示例：在Cambricon MLU上执行张量加法 C = A + B

// 1. 准备设备和张量描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_BANG, 0);

// 假设我们有两个形状为{1024, 1024}的FP16张量
std::vector<int64_t> shape = {1024, 1024};
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F16, shape.size(), shape.data());
infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F16, shape.size(), shape.data());
infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F16, shape.size(), shape.data());

// 2. 创建加法操作描述符
op::add::bang::Descriptor* add_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {a_desc, b_desc};
infiniStatus_t status = op::add::bang::Descriptor::create(
    handle, &add_desc, c_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（例如数据类型不匹配或形状不一致）
}

// 3. 分配设备内存和工作空间
size_t workspace_size = add_desc->workspaceSize();
void *d_a, *d_b, *d_c, *workspace;
cnrtMalloc(&d_a, 1024 * 1024 * sizeof(half));
cnrtMalloc(&d_b, 1024 * 1024 * sizeof(half));
cnrtMalloc(&d_c, 1024 * 1024 * sizeof(half));
cnrtMalloc(&workspace, workspace_size);

// 拷贝输入数据到设备（用户代码）
// cnrtMemcpy(d_a, h_a, ..., CNRT_MEM_TRANS_DIR_HOST2DEV);
// cnrtMemcpy(d_b, h_b, ..., CNRT_MEM_TRANS_DIR_HOST2DEV);

// 4. 创建执行队列
cnrtQueue_t queue;
cnrtQueueCreate(&queue);

// 5. 执行加法计算
std::vector<const void*> input_ptrs = {d_a, d_b};
status = add_desc->calculate(workspace, workspace_size, d_c, input_ptrs, queue);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（例如工作空间不足）
}

// 6. 同步并取回结果（可选）
cnrtQueueSync(queue);
// cnrtMemcpy(h_c, d_c, ..., CNRT_MEM_TRANS_DIR_DEV2HOST);

// 7. 清理资源
cnrtQueueDestroy(queue);
cnrtFree(d_a); cnrtFree(d_b); cnrtFree(d_c); cnrtFree(workspace);
delete add_desc;
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### Memory Management
- **NRAM分块策略**: NR AM（Near RAM）是MLU的高速片上内存，本实现采用分块（tiling）策略将大型张量分解为适合NRAM的小批次。批次大小计算公式：`max_batch = (NRAM_MAX_SIZE - ALIGN_SIZE*(N+1)) / ((N+1)*sizeof(Tdata))`，其中N=2为输入数量。对于FP32数据，若NRAM_MAX_SIZE为256KB，ALIGN_SIZE为128B，则max_batch约为64KB。
- **工作空间布局**: 设备工作空间按顺序存储：[输入指针数组(N*sizeof(void*))][元数据起始地址]。元数据区域依次包含：输出形状(ndim*size_t)、输出步长(ndim*ptrdiff_t)、所有输入形状(N*ndim*size_t)、所有输入步长(N*ndim*ptrdiff_t)、连续性标志(N*sizeof(bool))、广播标志(N*sizeof(bool))。
- **对齐优化**: NRAM缓冲区使用ALIGN_SIZE对齐（通常为128字节），通过`aligned_buf = (nram_buf + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1)`计算对齐地址，确保内存访问效率。

### Concurrency
- **MLU并行模型**: 使用BANG的2级并行层次：集群级（cluster）和核心级（core）。内核启动配置`dim.x = core_per_cluster, dim.y = cluster_count`，例如对于4核心/集群、16集群的MLU，共启动64个并行任务。
- **负载均衡**: 每个任务处理连续的元素范围，通过`elements_per_task = (output_size + taskDim - 1) / taskDim`计算，最后几个任务可能处理较少元素。
- **内存同步**: 使用`__sync_io()`确保所有异步内存拷贝完成，使用`__sync_compute()`确保计算完成，避免数据竞争。主机端使用`cnrtQueueSync(queue)`等待队列中所有操作完成。
- **函数类型选择**: 对于大规模连续操作（output_size > 1M），使用`CNRT_FUNC_TYPE_UNION1`内核类型，允许跨集群协作以提高性能。

### Performance
- **硬件加速**: 对于支持的浮点类型（half/bfloat16_t/float），直接调用Cambricon BANG库的`__bang_add`函数，该函数使用MLU的张量核心进行向量化加法，理论性能可达数十TOPS。
- **连续内存优化**: 对于连续张量，使用`__memcpy_async`进行大批量GDRAM↔NRAM拷贝，充分利用MLU的内存带宽（通常>300GB/s）。对于非连续张量，使用`calculateChunkSize`检测连续块并进行分段拷贝，减少小批量传输。
- **计算-内存重叠**: 通过异步内存拷贝（`__memcpy_async`）和显式同步（`__sync_io`），实现计算与内存传输的重叠，提高设备利用率。
- **时间复杂度**: O(n)，其中n为输出元素数量。在理想情况下（连续内存、足够并行度），可达到接近内存带宽限制的吞吐量。
- **空间复杂度**: O(n)用于输入/输出存储，加上O(1)的NRAM临时空间（不随问题规模增长）。

### Error Handling
- **数据类型验证**: 在`Descriptor::create`中使用`CHECK_DTYPE`宏确保输出类型为F16/BF16/F32，否则返回`INFINI_STATUS_BAD_TENSOR_DTYPE`。
- **形状一致性检查**: 使用`CHECK_SAME_SHAPE`宏验证三个张量形状完全一致，不支持广播（与某些框架如NumPy不同），错误时返回相应状态码。
- **工作空间验证**: `calculate`方法检查`workspace_size >= _workspace_size`，不足时返回`INFINI_STATUS_INSUFFICIENT_WORKSPACE`。
- **CNRT错误传播**: 所有CNRT API调用（如`cnrtMemcpy`、`cnrtQueueSync`）都通过`CNRT_CHECK`宏包装，失败时直接返回错误状态。
- **Result类型**: 基础设施使用`utils::Result<T>`类型安全的错误处理，避免异常开销。例如`ElementwiseInfo::create`返回`Result<ElementwiseInfo>`，成功时包含有效对象，失败时包含错误码。

### Dependencies
- **BANG SDK**: 依赖Cambricon BANG编程环境，包括：
  - `cnnl.h`: Cambricon CNNL库（机器学习加速原语）
  - `cnrt.h`: Cambricon CNRT运行时API，提供设备管理、内存管理、内核启动等功能
  - `__bang_add`: BANG内置函数，执行张量加法
  - `__memcpy_async`: 异步内存拷贝函数
  - `__nram__`: 关键字，声明NRAM上的变量
  - `__mlu_device__` / `__mlu_global__`: 函数标记，分别表示设备端可调用和全局内核入口
- **基础设施模块**:
  - `op::elementwise::ElementwiseInfo`: 元数据管理，提供形状、步长、连续性等信息
  - `device::bang::Handle::Internal`: MLU设备句柄，提供硬件信息（核心数、集群数）
  - `op::elementwise::bang::DeviceImpl`: 通用逐元素操作的设备实现
  - `InputIndexer`: 输入索引计算辅助类，处理广播和非连续布局
  - `nonContiguousMemcpy`: 非连续内存拷贝模板函数
  - `indexToOffset`, `calculateChunkSize`: 索引计算辅助函数
- **编译器**: 需要Cambricon BANG编译器（类似nvcc的MLU专用编译器），支持.mlu文件的编译。

### Design Patterns
- **Pimpl (Pointer to Implementation)**: `DeviceImpl`使用`std::shared_ptr<Opaque>`隐藏实现细节，减少编译依赖并允许动态调整。
- **Strategy Pattern**: `AddOp`作为可调用对象（仿函数），实现统一的逐元素操作接口，使同一内核框架支持不同操作（加、减、乘等）。
- **Template Method**: `ElementwiseInfo::create`定义元数据构造的模板流程，具体操作通过宏参数化。
- **Macro Metaprogramming**: 大量使用宏（`ELEMENTWISE_DESCRIPTOR`, `CREATE_ELEMENTWISE_BANG_DESCRIPTOR`, `LAUNCH_ELEMENTWISE_KERNEL`等）生成重复代码，减少手写错误并保持接口一致性。
- **CRTP (Curiously Recurring Template Pattern)**: 间接体现，设备实现通过模板参数`Op`静态分发，实现编译期多态，避免虚函数开销。
- **RAII (Resource Acquisition Is Initialization)**: 使用`std::unique_ptr`和`std::shared_ptr`管理资源（如`DeviceImpl`），确保异常安全。
- **Kernel Fusion**: 虽然本模块未实现，但架构支持未来扩展，例如将加法和激活函数融合为一个内核。
