# `Kunlun Rearrange Operator` Core Implementation Documentation

本模块实现了在昆仑（Kunlun）XPU设备上的张量重排操作（Rearrange Operator），支持任意维度的张量reshape、transpose和stride变换操作。该实现通过XPU内核利用Local Memory（LM）进行数据缓存优化，实现了高效的异构内存拷贝和索引转换。

## 1. Module Structure

- **`rearrange_kunlun.h`**: 定义核心数据结构`RearrangeInfo`和公共API类`Descriptor`，包含张量元信息验证和工作空间大小计算
- **`rearrange_kunlun.xpu`**: 实现XPU内核函数`rearrangeKernel`和CUDA风格的kernel launch逻辑，包含设备内存管理和异步数据传输

## 2. Core Classes

### `RearrangeInfo`
- **Location**: `rearrange_kunlun.h`
- **Primary Function**: 封装张量重排操作的元数据，包括输入/输出张量的shape、stride和数据类型，并计算所需的设备工作空间大小
- **Key Members**:
  - `shape: std::vector<size_t>`: 张量形状（输出和输入形状必须相同）
  - `src_strides: std::vector<ptrdiff_t>`: 源张量的步长信息（每维度的字节跨度）
  - `dst_strides: std::vector<ptrdiff_t>`: 目标张量的步长信息
  - `dtype: infiniDtype_t`: 张量数据类型（支持F32、BF16、F16）
  - `workspace_size: size_t`: 设备端工作空间大小（存储shape和stride数组）
- **Core Methods**:
  - `nelements()`: 计算张量总元素数，使用`std::accumulate`对shape数组求乘积，O(n)时间复杂度
  - `ndim()`: 返回张量维度数
  - `workspaceSize()`: 返回工作空间大小，计算公式为`sizeof(size_t)*ndim + sizeof(ptrdiff_t)*ndim*2`
  - `create(y_desc, x_desc)`: 静态工厂方法，验证输入/输出描述符的dtype和ndim一致性，检查shape是否匹配，计算并返回`Result<RearrangeInfo>`
- **Lifecycle**: 值类型，通过`create()`静态方法构造，失败时返回错误状态码

### `Descriptor`
- **Location**: `rearrange_kunlun.h` (声明), `rearrange_kunlun.xpu` (实现)
- **Primary Function**: Kunlun设备的重排操作描述符，继承自`InfiniopDescriptor`，管理设备工作空间和内核执行
- **Key Members**:
  - `_opaque: Opaque*`: 不透明指针，指向包含`Handle::Internal`共享指针和设备workspace的内部结构
  - `_info: RearrangeInfo`: 不可变的张量元信息
- **Core Methods**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态创建方法，调用`RearrangeInfo::create()`验证输入，在L3缓存（`XPU_MEM_L3`）分配workspace，构造Descriptor实例
  - `calculate(y, x, stream)`: 执行重排操作，通过异步memcpy将shape/stride传输到设备，启动模板化kernel
  - `~Descriptor()`: 析构函数，删除`_opaque`指针触发`Opaque`析构，自动释放设备workspace
- **Lifecycle**:
  1. **Construction**: 通过`create()`静态方法分配，在L3内存分配workspace
  2. **Execution**: 可多次调用`calculate()`，每次复用同一workspace但更新shape/stride数据
  3. **Destruction**: 析构时调用`xpu_free()`释放设备workspace

### `Descriptor::Opaque`
- **Location**: `rearrange_kunlun.xpu`
- **Primary Function**: Pimpl（Pointer to Implementation）模式的实现类，封装设备相关资源和RAII管理
- **Key Members**:
  - `internal: std::shared_ptr<device::kunlun::Handle::Internal>`: Kunlun设备句柄的内部状态，保持引用计数
  - `workspace: void*`: 设备端工作空间指针，存储shape、src_strides、dst_strides数组
- **Core Methods**:
  - `~Opaque()`: 析构函数，检查workspace非空后调用`xpu_free()`释放L3内存

### `rearrangeKernel<BUFF_SIZE, Tdata>`
- **Location**: `rearrange_kunlun.xpu`
- **Primary Function**: XPU设备端内核函数，实现基于Local Memory缓存的重排操作，采用两阶段访存模式（GM→LM→GM）优化带宽
- **Template Parameters**:
  - `BUFF_SIZE: unsigned int`: Local Memory缓冲区大小（固定为64元素）
  - `Tdata`: 数据类型（float、bfloat16_t、half）
- **Kernel Parameters**:
  - `y: Tdata*`: 输出张量的全局内存指针
  - `x: const Tdata*`: 输入张量的全局内存指针
  - `shape/x_stride/y_stride: const void*`: 形状和步长数组的设备指针
  - `ndim/total_size: uint32_t`: 张量维度数和总元素数
- **Core Algorithm**:
  1. **线程分区**: 通过`core_id()`, `cluster_id()`计算全局线程ID，采用12 clusters × 64 cores布局（`<<<12, 64>>>`）
  2. **LM预加载**: 使用`GM2LM_ASYNC`异步拷贝shape/stride到Local Memory，调用`mfence()`等待完成
  3. **分块处理**: 每个线程处理`len_per_loop = min(BUFF_SIZE, roundup_div(total_size, nthreads))`个元素
  4. **双阶段传输**:
     - **阶段1（GM→LM）**: 循环调用`indexToOffset()`计算源偏移量，异步读取输入数据到`x_local`缓冲区
     - **阶段2（LM→GM）**: 循环调用`indexToOffset()`计算目标偏移量，异步写入数据到输出张量
  5. **集群同步**: 每个分块结束调用`sync_cluster()`同步cluster内所有cores
- **Thread Mapping**: Kunlun XPU采用cluster-based架构，`thread_id = ncores * cluster_id() + cid`，支持多级并行

### `launchKernel<BUFF_SIZE>()`
- **Location**: `rearrange_kunlun.xpu`
- **Primary Function**: 模板化的kernel启动函数，根据dtype分发特化版本，解析workspace内存布局
- **Algorithm**:
  1. 计算workspace内三个数组的设备指针：
     - `d_shape`: 起始位置
     - `d_src_strides`: `d_shape + ndim`
     - `d_dst_strides`: `d_src_strides + ndim`
  2. 使用switch-case匹配dtype，展开`LAUNCH_KERNEL`宏
  3. 调用XPU kernel launch语法：`rearrangeKernel<BUFF_SIZE, Tdata><<<12, 64, stream>>>(...)`
- **Supported Types**: F32 (float), BF16 (bfloat16_t), F16 (half)

## 3. API Interface

```cpp
// 创建重排操作描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,              // Kunlun设备句柄
    Descriptor **desc_ptr,                // [out] 输出的描述符指针
    infiniopTensorDescriptor_t y_desc,    // 输出张量描述符
    infiniopTensorDescriptor_t x_desc     // 输入张量描述符
);
// 返回值: SUCCESS / BAD_TENSOR_DTYPE / BAD_TENSOR_SHAPE / 内存分配失败

// 执行重排计算
infiniStatus_t Descriptor::calculate(
    void *y,              // 输出张量的设备内存指针
    const void *x,        // 输入张量的设备内存指针
    void *stream          // Kunlun流句柄
) const;
// 返回值: SUCCESS / BAD_TENSOR_DTYPE / 异步拷贝失败
```

## 4. Usage Example

```cpp
// 示例: 在Kunlun XPU上执行(2, 3)矩阵的转置操作
#include "rearrange_kunlun.h"

using namespace op::rearrange::kunlun;

// 1. 准备张量描述符 (假设shape为[2, 3])
int64_t shape[] = {2, 3};
int64_t src_strides[] = {3, 1};  // C contiguous
int64_t dst_strides[] = {1, 2};  // Fortran contiguous (转置)
auto x_desc = createTensorDescriptor(handle, 2, shape, src_strides, INFINI_DTYPE_F32);
auto y_desc = createTensorDescriptor(handle, 2, shape, dst_strides, INFINI_DTYPE_F32);

// 2. 创建重排操作描述符 (分配workspace并验证元数据)
Descriptor* rearrange_desc = nullptr;
infiniStatus_t status = Descriptor::create(handle, &rearrange_desc, y_desc, x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误 (dtype不匹配、shape不一致、L3内存不足等)
}

// 3. 分配设备内存并初始化数据
float* d_x, *d_y;
xpu_malloc(&d_x, 6 * sizeof(float), XPU_MEM_HBM);
xpu_malloc(&d_y, 6 * sizeof(float), XPU_MEM_HBM);
// ... 通过xpu_memcpy_async上传数据到d_x ...

// 4. 执行重排操作 (内部会先拷贝shape/stride到workspace，再启动kernel)
kunlunStream_t stream = reinterpret_cast<kunlunStream_t>(handle->stream);
status = rearrange_desc->calculate(d_y, d_x, stream);

// 5. 等待完成并下载结果
xpu_stream_synchronize(stream);
xpu_memcpy_async(h_y, d_y, 6 * sizeof(float), XPU_DEVICE_TO_HOST, stream);

// 6. 清理资源 (析构函数自动释放workspace)
delete rearrange_desc;
xpu_free(d_x);
xpu_free(d_y);
```

## 5. Implementation Details

### 内存管理 (Memory Management)
- **Workspace分配**: 使用`xpu_malloc(workspace_size, XPU_MEM_L3)`在L3缓存分配设备内存，存储shape（`size_t[ndim]`）和stride数组（`ptrdiff_t[ndim*2]`）
- **内存布局**: Workspace内三段连续布局：shape → src_strides → dst_strides，通过指针偏移访问
- **RAII模式**: `Opaque`析构函数自动调用`xpu_free(workspace)`，防止内存泄漏

### 并发控制 (Concurrency)
- **Kernel配置**: 固定使用12个cluster，每个cluster 64个core（总共768个线程）
- **数据分区**: 每个线程处理`len_per_loop = min(64, ceil_div(total_size, 768))`个元素，采用stride循环模式（`start += nthreads * len_per_loop`）
- **同步机制**:
  - `mfence()`: 等待GM2LM异步拷贝完成
  - `sync_cluster()`: 同步cluster内所有core（在LM2GM写入后调用）

### 性能优化 (Performance)
- **Local Memory复用**: 64元素的`x_local`缓冲区，减少GM访问次数（理论带宽提升2倍）
- **异步拷贝流水线**: `GM2LM_ASYNC`和`LM2GM_ASYNC`隐藏内存延迟
- **两级并行**: Cluster级（12个）+ Core级（每个cluster 64个）充分利用XPU架构
- **索引计算优化**: 调用`indexToOffset()`函数（定义在`kunlun_kernel_common.h`），编译器内联后减少指令数

### 错误处理 (Error Handling)
- **类型检查**: `RearrangeInfo::create()`验证x和y的dtype、ndim必须一致，返回`INFINI_STATUS_BAD_TENSOR_DTYPE`或`INFINI_STATUS_BAD_TENSOR_SHAPE`
- **形状验证**: 使用`CHECK_SAME_SHAPE(x_shape, y_shape)`宏确保输入输出shape相同
- **设备操作检查**: 使用`CHECK_KUNLUN()`宏包装xpu API调用，失败时提前返回错误码
- **不支持的dtype**: `launchKernel()`的switch default分支返回`INFINI_STATUS_BAD_TENSOR_DTYPE`

### 依赖关系 (Dependencies)
- **外部依赖**:
  - `device::kunlun::Handle`: Kunlun设备管理器（定义在`kunlun_handle.h`）
  - `device::kunlun::kernel::indexToOffset()`: 索引转换工具函数（定义在`kunlun_kernel_common.h`）
  - XPU Driver API: `xpu_malloc()`, `xpu_free()`, `xpu_memcpy_async()`, `xpu_stream_synchronize()`
- **内部依赖**:
  - `infiniopTensorDescriptor`: 张量描述符基类（提供`dtype()`, `ndim()`, `shape()`, `strides()`方法）
  - `InfiniopDescriptor`: 操作描述符基类（存储`device_type`和`device_id`）

### 设计模式 (Design Patterns)
- **Pimpl (Pointer to Implementation)**: `Descriptor`通过`Opaque`指针隐藏设备相关实现，减少头文件依赖
- **RAII (Resource Acquisition Is Initialization)**: `Opaque`析构函数自动释放workspace
- **Factory Method**: `Descriptor::create()`和`RearrangeInfo::create()`静态工厂方法封装创建逻辑
- **Template Method**: `launchKernel<BUFF_SIZE>()`通过模板特化避免代码重复
- **Strategy Pattern**: 通过switch-case根据dtype选择不同的kernel实例化策略
