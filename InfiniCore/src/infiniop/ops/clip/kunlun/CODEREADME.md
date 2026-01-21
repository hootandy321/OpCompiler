# Kunlun CLIP Operation Core Implementation Documentation

该模块实现了在昆仑（XPU）设备上的 CLIP（裁剪）操作，这是一个逐元素(elementwise)操作，将张量值裁剪到指定范围 [min, max] 内。该实现基于昆仑 XPU 的异构计算架构，利用本地内存(local memory)和全局内存(global memory)的分层存储结构进行高性能并行计算。

## 1. Module Structure

- **`clip_kunlun.h`**: CLIP 操作的 API 声明文件，定义描述符(descriptor)宏和公共接口
- **`clip_kunlun.xpu`**: CLIP 操作的核心实现文件，包含描述符的创建、计算调度和设备内存管理
- **`kernel.h`**: CLIP 核算子(kernel functor)定义，实现裁剪算法的设备端计算逻辑

## 2. Core Classes

### `op::clip::kunlun::Descriptor`
- **Location**: `clip_kunlun.xpu` (通过宏定义在 `clip_kunlun.h`)
- **Primary Function**: CLIP 操作的描述符类，负责管理操作的生命周期、验证参数、分配工作空间和调度内核执行
- **Key Members**:
  - `_dtype`: 支持的数据类型 (F16, F32, BF16)
  - `_info`: `ElementwiseInfo` 对象，存储输入/输出张量的形状、步幅、广播等元数据
  - `_device_info`: `DeviceImpl` 指针，封装昆仑设备实现细节
  - `_workspace_size`: 设备工作空间大小（存储元数据 + 输入指针数组）
  - `_device`, `_device_id`: 昆仑设备标识符
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，验证输入参数并初始化描述符
    - **验证逻辑**: 检查数据类型为 F16/F32/BF16，验证输入/输出/min/max 四个张量形状完全一致
    - **内存分配**: 通过 `CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR` 宏计算工作空间大小并创建 `ElementwiseInfo` 和 `DeviceImpl`
    - **复杂度**: O(1)
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 CLIP 操作的核心方法
    - **工作空间验证**: 检查 `workspace_size >= _workspace_size`
    - **类型分发**: 根据 `_dtype` 分别调用模板实例化的设备计算函数
    - **模板参数**: BLOCK_SIZE=8（每块线程数），Op=ClipOp，Tdata=half/bfloat16_t/float
    - **复杂度**: O(n)，其中 n 为输出张量元素个数
- **Lifecycle**:
  - 由 `create()` 静态方法构造，分配在堆内存
  - 析构函数为默认实现（`= default`）
  - 依赖父级 `device::kunlun::Handle` 管理设备资源生命周期

### `op::clip::kunlun::ClipOp`
- **Location**: `kernel.h`
- **Primary Function**: 设备端函数对象(functor)，定义 CLIP 操作的单元素计算逻辑
- **Key Members**:
  - `num_inputs`: 编译时常量，值为 3（对应输入 x、最小值 min、最大值 max）
- **Core Methods**:
  - `operator()(const T *inputs) const`: 模板函数，通用类型的裁剪计算
    - **算法**: `fmax(fmin(x, max_val), min_val)`，先应用上界约束再应用下界约束
    - **参数**: inputs[0]=x, inputs[1]=min_val, inputs[2]=max_val
    - **返回值**: 裁剪后的值
    - **复杂度**: O(1)
  - `operator()(const bfloat16_t *inputs) const`: BF16 特化版本
    - **精度保护**: 将 BF16 转换为 float 进行计算（`__bfloat162float`），避免精度损失
    - **计算流程**: BF16 → Float → fmax/fmin → Float → BF16（`__float2bfloat16`）
    - **复杂度**: O(1)
- **Lifecycle**:
  - 编译期静态结构体，无需实例化
  - 通过模板参数传递给 `elementwiseKernel`

### `op::elementwise::kunlun::DeviceImpl`
- **Location**: 依赖的父级模块 `elementwise_kunlun.h`
- **Primary Function**: 昆仑设备上的逐元素操作执行引擎，提供内存管理和内核启动功能
- **Key Members**:
  - `_opaque`: PIMPL 指针，隐藏实现细节（`std::shared_ptr<Opaque>`）
  - `Opaque::internal`: 昆仑设备句柄的内部状态（`std::shared_ptr<device::kunlun::Handle::Internal>`）
- **Core Methods**:
  - `create(args...)`: 静态工厂方法，创建 `DeviceImpl` 实例
  - `calculate<BLOCK_SIZE, Op, Tdata>(info, workspace, output, inputs, stream)`: 模板方法调度计算
    - 解析输入数量 `N = Op::num_inputs`
    - 转发到 `_opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>`
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(...)`: 调用 `launchElementwiseKernel` 启动内核
  - `infoToDevice<N>(...)`: 将主机端元数据复制到设备工作空间
    - **内存布局**: [输入指针数组 (N*sizeof(void*))] [输出形状 (ndim*sizeof(size_t))] [输出步幅 (ndim*sizeof(ptrdiff_t))] [输入形状 (N*ndim*sizeof(size_t))] [输入步幅 (N*ndim*sizeof(ptrdiff_t))] [连续标志 (N*sizeof(bool))] [广播标志 (N*sizeof(bool))]
    - **异步传输**: 使用 `xpu_memcpy_async` 进行 H2D 异步拷贝
  - `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout>(...)`: 启动 XPU 内核
    - **线程配置**: `<<<BLOCK_SIZE, 64, stream>>>`（BLOCK_SIZE 个 cluster，每个 cluster 64 个计算单元）
    - **内核参数**: 元数据指针、输入/输出数据指针、用户自定义参数
- **Lifecycle**:
  - 通过 `create()` 静态方法构造，返回 `Result<DeviceImpl*>`
  - 析构函数为默认实现，通过 `shared_ptr` 自动管理资源

## 3. API Interface

```cpp
// 公共 API：创建 CLIP 操作描述符
infiniStatus_t op::clip::kunlun::Descriptor::create(
    infiniopHandle_t handle_,              // 昆仑设备句柄
    Descriptor **desc_ptr,                 // 输出：描述符指针
    infiniopTensorDescriptor_t out_desc,   // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量 [x, min, max]
);
// 返回值: INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / 形状不匹配错误

// 公共 API：执行 CLIP 计算
infiniStatus_t op::clip::kunlun::Descriptor::calculate(
    void *workspace,                       // 设备工作空间指针
    size_t workspace_size,                 // 工作空间大小（字节）
    void *output,                          // 输出数据指针（设备内存）
    std::vector<const void *> inputs,      // 输入数据指针向量 [x, min, max]（设备内存）
    void *stream                           // 昆仑流句柄
) const;
// 返回值: INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. Usage Example

```cpp
// 示例：在昆仑 XPU 上执行 CLIP 操作
#include "clip_kunlun.h"

// 1. 初始化昆仑设备和句柄
infiniopHandle_t handle;
infiniopCreate(&handle, device_id);

// 2. 准备输入/输出张量描述符（假设形状为 {1024, 1024}）
std::vector<int64_t> shape = {1024, 1024};
infiniopTensorDescriptor_t x_desc, min_desc, max_desc, out_desc;
infiniopCreateTensor(handle, &x_desc, INFINI_DTYPE_F16, shape.size(), shape.data());
infiniopCreateTensor(handle, &min_desc, INFINI_DTYPE_F16, shape.size(), shape.data());
infiniopCreateTensor(handle, &max_desc, INFINI_DTYPE_F16, shape.size(), shape.data());
infiniopCreateTensor(handle, &out_desc, INFINI_DTYPE_F16, shape.size(), shape.data());

// 3. 创建 CLIP 操作描述符
op::clip::kunlun::Descriptor *clip_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {x_desc, min_desc, max_desc};
auto status = op::clip::kunlun::Descriptor::create(handle, &clip_desc, out_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（数据类型不匹配或形状不一致）
}

// 4. 分配设备内存和工作空间
size_t workspace_size = clip_desc->getWorkspaceSize();  // 获取所需工作空间大小
void *d_x, *d_min, *d_max, *d_out, *d_workspace;
xpu_malloc(&d_x, 1024 * 1024 * sizeof(half));
xpu_malloc(&d_min, 1024 * 1024 * sizeof(half));
xpu_malloc(&d_max, 1024 * 1024 * sizeof(half));
xpu_malloc(&d_out, 1024 * 1024 * sizeof(half));
xpu_malloc(&d_workspace, workspace_size);

// 5. 将输入数据从主机传输到设备
xpu_memcpy_async(d_x, h_x, 1024 * 1024 * sizeof(half), XPU_HOST_TO_DEVICE, stream);
xpu_memcpy_async(d_min, h_min, 1024 * 1024 * sizeof(half), XPU_HOST_TO_DEVICE, stream);
xpu_memcpy_async(d_max, h_max, 1024 * 1024 * sizeof(half), XPU_HOST_TO_DEVICE, stream);

// 6. 执行 CLIP 计算
std::vector<const void *> input_ptrs = {d_x, d_min, d_max};
status = clip_desc->calculate(d_workspace, workspace_size, d_out, input_ptrs, stream);

// 7. 将结果传回主机
half h_out[1024 * 1024];
xpu_memcpy_async(h_out, d_out, 1024 * 1024 * sizeof(half), XPU_DEVICE_TO_HOST, stream);
xpu_stream_synchronize(stream);

// 8. 清理资源
delete clip_desc;
xpu_free(d_x); xpu_free(d_min); xpu_free(d_max); xpu_free(d_out); xpu_free(d_workspace);
infiniopDestroy(handle);
```

## 5. Implementation Details

### Memory Management
- **分层存储架构**:
  - **全局内存 (GM)**: 存储输入/输出张量数据和元数据，使用 `__global_ptr__` 修饰符访问
  - **本地内存 (LM)**: 每个计算单元的私有高速缓存，使用 `__local__` 修饰符，用于缓存输入元素和形状/步幅元数据
  - **工作空间布局**: 设备端工作空间分为两部分：输入指针数组（N * sizeof(void*)） + 元数据区域（形状、步幅、广播标志等），总大小由 `ElementwiseInfo::getMetaMemSize() + N * sizeof(void*)` 计算

- **异步数据传输**:
  - 使用 `GM2LM_ASYNC` 和 `LM2GM_ASYNC` 宏进行全局内存与本地内存间的异步拷贝
  - 使用 `xpu_memcpy_async` 进行主机到设备的异步内存传输
  - 通过 `mfence()` 内存屏障确保异步拷贝完成后再进行计算

- **内存访问优化**:
  - **连续张量优化**: 当输入/输出张量连续时，直接使用线性索引，避免 `indexToOffset` 计算（`getOutputIndex` 函数）
  - **输入索引器**: `InputIndexer` 结构体封装输入张量的索引计算，支持广播和非连续张量的步幅映射
  - **本地内存缓存**: 将形状、步幅、输入指针等元数据缓存在本地内存，减少全局内存访问

### Concurrency
- **XPU 线程模型**:
  - **三级并行**: Cluster（计算簇） → Core（计算核） → Thread（线程）
  - **线程索引计算**: `thread_id = ncores * cluster_id() + cid`，其中 `cid` 为核心 ID，`ncores` 为每簇核心数
  - **并行策略**: 将输出元素按块分配给线程，每个线程处理 `BUFF_SIZE=64` 个元素的块（`len_per_loop = min(64, roundup_div(output_size, nthreads))`）

- **同步机制**:
  - **簇内同步**: 使用 `mfence()` 确保本地内存异步拷贝完成
  - **簇间同步**: 使用 `sync_cluster()` 确保所有簇的计算完成后才结束内核
  - **流同步**: 主机端调用 `xpu_stream_synchronize(stream)` 等待操作完成

- **线程安全**:
  - 描述符对象在创建后不可变（immutable），多线程可安全并发调用 `calculate`（需不同的 workspace 和 stream）
  - 设备句柄内部状态通过 `std::shared_ptr` 管理引用计数，确保资源安全释放

### Performance
- **算法复杂度**:
  - **时间复杂度**: O(n)，其中 n 为输出张量元素个数，每个元素仅需两次浮点比较（fmin + fmax）
  - **空间复杂度**: O(m)，其中 m 为元数据大小（形状、步幅、标志等），与数据大小无关
  - **并行度**: 理论上可并行处理所有元素，受限于 XPU 设备的 cluster 数和核心数

- **内核启动配置**:
  - **Block Size**: BLOCK_SIZE=8（模板参数），控制启动的 cluster 数量
  - **Cluster Size**: 64 个计算单元/cluster（硬件配置）
  - **线程总数**: `BLOCK_SIZE * 64`，通常小于等于设备的物理核心数

- **优化技术**:
  - **循环展开**: 使用 `#pragma unroll` 完全展开输入拷贝循环（N 个输入）
  - **本地内存聚合**: 将多个输入元素的元数据（形状、步幅）批量加载到本地内存
  - **分支消除**: 通过模板特化避免类型判断分支（half/float/bfloat16_t 分别实例化）
  - **向量化友好**: 逐元素操作无依赖关系，易于编译器自动向量化

### Error Handling
- **参数验证**:
  - 使用 `CHECK_DTYPE` 宏验证数据类型为 F16/F32/BF16
  - 使用 `CHECK_SAME_SHAPE` 宏验证输入/min/max/output 四个张量形状完全一致
  - 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE` 或 `INFINI_STATUS_BAD_TENSOR_SHAPE` 错误码

- **运行时检查**:
  - 在 `calculate` 方法中检查工作空间大小是否充足（`workspace_size < _workspace_size` 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`）
  - 内核启动前检查输出大小是否为 0（提前返回成功）
  - 线程索引越界保护（`if (cid >= ncores) return;`）

- **错误传播**:
  - 使用 `CHECK_RESULT` 和 `CHECK_STATUS` 宏检查 `Result<T>` 类型的返回值
  - 使用 `CHECK_KUNLUN` 宏检查昆仑 XPU API 调用状态（`xpu_memcpy_async` 等）
  - 错误码通过 `infiniStatus_t` 枚举返回给调用者

### Dependencies
- **核心依赖**:
  - `elementwise_kunlun_api.h`: 提供 `ELEMENTWISE_DESCRIPTOR` 宏和 `DeviceImpl` 类
  - `elementwise_kunlun.h`: 提供逐元素操作的内核启动和内存管理基础设施
  - `kunlun_handle.h`: 昆仑设备句柄和资源管理
  - `kunlun_kernel_common.h`: 昆仑内核通用工具函数（`indexToOffset`, `roundup_div` 等）
  - `xpu/kernel/xtdk_io.h`: 昆仑 XPU 内核 I/O 原语（`__global_ptr__`, `__local__`, `GM2LM_ASYNC`, `mfence`）

- **外部依赖**:
  - 昆仑 XPU 驱动运行时（libxpu-runtime.so）
  - XPU 编译器（支持 `.xpu` 文件编译）
  - InfiniOp 框架基础设施（张量描述符、错误处理、工具类）

### Design Patterns
- **PIMPL (Pointer to Implementation)**: `DeviceImpl` 使用 `Opaque` 内部结构体隐藏实现细节，减少头文件依赖和编译耦合
- **CRTP (Curiously Recurring Template Pattern)**: 通过 `ELEMENTWISE_DESCRIPTOR(clip, kunlun)` 宏生成描述符类，复用逐元素操作的通用代码
- **Template Method**: `calculate` 方法定义算法骨架，类型分发通过模板特化实现
- **Factory Pattern**: `create()` 静态工厂方法封装对象创建逻辑，返回 `Result<Descriptor*>` 类型安全的错误处理
- **Strategy Pattern**: `ClipOp` 函数对象封装裁剪算法，可通过模板参数替换为其他逐元素操作
- **RAII (Resource Acquisition Is Initialization)**: 使用 `std::shared_ptr` 管理设备句柄和内部状态，自动释放资源
- **Barton-Nackman Trick**: `ELEMENTWISE_DESCRIPTOR` 宏在基类中定义派生类的接口，实现静态多态
