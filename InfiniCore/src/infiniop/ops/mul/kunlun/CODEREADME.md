# `Infiniop Mul Kunlun` 逐元素乘法运算实现文档

本模块实现了昆仑（Kunlun）XPU 设备上的逐元素乘法（Element-wise Multiplication）算子，支持 FP16、FP32 和 BF16 三种数据类型，作为 Infiniop 算子库中 mul 操作的昆仑后端实现。

## 1. 模块结构

- **`kernel.h`**: 定义乘法运算的核心算子（MulOp），包含通用模板和 bfloat16_t 特化版本
- **`mul_kunlun.h`**: 通过 ELEMENTWISE_DESCRIPTOR 宏定义 Descriptor 类的接口声明
- **`mul_kunlun.xpu`**: 实现乘法描述符的创建（create）和计算（calculate）方法

## 2. 核心类与算子

### `MulOp` 算子函数对象
- **位置**: `kernel.h`
- **主要功能**: 定义逐元素乘法运算的 CUDA/XPU 设备端计算逻辑
- **核心成员**:
  - `num_inputs`: 静态常量，固定为 2（表示二元运算）
- **核心方法**:
  - `operator()(const T *inputs) const`: 通用模板版本
    - 从 inputs 数组中读取两个操作数 `inputs[0]` 和 `inputs[1]`
    - 执行乘法运算 `return a * b`
    - 时间复杂度: O(1)
  - `operator()(const bfloat16_t *inputs) const`: bfloat16_t 特化版本
    - 使用 `__bfloat162float` 将 BF16 转换为 float 以提高计算精度
    - 在 float 域执行乘法：`a_f * b_f`
    - 使用 `__float2bfloat16` 将结果转换回 BF16
    - 避免了低精度 BF16 直接乘法的精度损失
- **生命周期**: 函数对象，无状态，在设备端调用时直接实例化

### `Descriptor` 类
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(mul, kunlun)` 宏在 `mul_kunlun.h` 中定义，实现在 `mul_kunlun.xpu`
- **主要功能**: 管理乘法操作的元数据、工作空间和设备端实现
- **核心成员**:
  - `_dtype`: 存储输出张量的数据类型（`infiniDtype_t`）
  - `_info`: `op::elementwise::ElementwiseInfo` 对象，封装输入/输出张量的形状、步长、广播等信息
  - `_device_info`: `op::elementwise::kunlun::DeviceImpl` 智能指针，负责实际的设备端 kernel 启动
  - `_workspace_size`: 设备端所需工作空间大小（字节），用于存储元数据和输入指针数组
- **核心方法**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法
    - **参数验证**:
      - 检查数据类型是否为 FP16/F32/BF16（通过 `CHECK_DTYPE` 宏）
      - 验证输入和输出形状完全一致（通过 `CHECK_SAME_SHAPE` 宏）
    - **元数据构造**:
      - 调用 `ElementwiseInfo::create` 生成操作元数据（形状、步长、广播标志等）
      - 创建 `DeviceImpl` 实例
      - 计算工作空间大小 = `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
    - **描述符实例化**: 使用 `CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR` 宏构造 Descriptor 对象
    - 返回 `INFINI_STATUS_SUCCESS` 或相应的错误码
    - 时间复杂度: O(N)，其中 N 为张量维度数

  - `calculate(workspace, workspace_size, output, inputs, stream) const`: 执行乘法计算
    - **工作空间检查**: 验证 `workspace_size >= _workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
    - **类型分发**: 根据 `_dtype` 调用对应的模板特化：
      - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<8, MulOp, half>(...)`
      - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<8, MulOp, bfloat16_t>(...)`
      - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<8, MulOp, float>(...)`
    - **Kernel 启动参数**:
      - 第一个模板参数 `8`: Kernel 启动的 BLOCK_SIZE（集群/线程块数量）
      - 第二个模板参数 `MulOp`: 设备端算子函数对象
      - 第三个模板参数: 数据类型（half/bfloat16_t/float）
    - 返回 `INFINI_STATUS_SUCCESS` 或错误码

- **生命周期**:
  - 在 `create` 方法中通过 `new Descriptor(...)` 构造
  - 析构函数在 `mul_kunlun.xpu` 中定义为 `default`
  - 由 Infiniop 框架管理生命周期，调用方负责释放

## 3. API 接口

```cpp
// 创建乘法描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                      // 昆仑设备句柄
    Descriptor **desc_ptr,                         // 输出：描述符指针的指针
    infiniopTensorDescriptor_t out_desc,           // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（大小为2）
);
// 返回值：成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码
// 前置条件：
//   - input_desc_vec.size() == 2
//   - out_desc 和所有 input_desc 的 dtype 必须是 F16/F32/BF16
//   - 所有张量的形状必须完全相同（不支持广播）

// 执行乘法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                    // 设备端工作空间指针
    size_t workspace_size,              // 工作空间大小（字节）
    void *output,                       // 输出张量的设备端指针
    std::vector<const void *> inputs,   // 输入张量的设备端指针向量（inputs[0] * inputs[1]）
    void *stream                        // 昆仑计算流
) const;
// 返回值：成功返回 INFINI_STATUS_SUCCESS
// 前置条件：
//   - workspace_size >= workspaceSize()
//   - output 和 inputs 中所有指针均指向已分配的设备内存
```

## 4. 使用示例

```cpp
// 示例：在昆仑 XPU 上执行逐元素乘法 C = A * B

#include "mul_kunlun.h"
#include "infiniop.h"

// 1. 准备张量描述符（假设形状为 [1024, 1024]）
std::vector<int64_t> shape = {1024, 1024};
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F16, shape.data(), shape.size(), nullptr);
infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F16, shape.data(), shape.size(), nullptr);
infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F16, shape.data(), shape.size(), nullptr);

// 2. 创建昆仑设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_KUNLUN, 0);

// 3. 创建乘法描述符
op::mul::kunlun::Descriptor *mul_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> input_descs = {a_desc, b_desc};
auto status = op::mul::kunlun::Descriptor::create(handle, &mul_desc, c_desc, input_descs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配设备内存和工作空间
half *d_a, *d_b, *d_c;
size_t tensor_size = 1024 * 1024 * sizeof(half);
xpu_malloc((void**)&d_a, tensor_size);
xpu_malloc((void**)&d_b, tensor_size);
xpu_malloc((void**)&d_c, tensor_size);

size_t workspace_size = mul_desc->workspaceSize();
void *d_workspace;
xpu_malloc(&d_workspace, workspace_size);

// 5. 将输入数据从主机复制到设备
xpu_memcpy_async(d_a, host_a, tensor_size, XPU_HOST_TO_DEVICE, stream);
xpu_memcpy_async(d_b, host_b, tensor_size, XPU_HOST_TO_DEVICE, stream);

// 6. 执行乘法计算
std::vector<const void *> inputs = {d_a, d_b};
status = mul_desc->calculate(d_workspace, workspace_size, d_c, inputs, stream);

// 7. 将结果复制回主机
xpu_memcpy_async(host_c, d_c, tensor_size, XPU_DEVICE_TO_HOST, stream);
xpu_stream_synchronize(stream);

// 8. 清理资源
delete mul_desc;
xpu_free(d_a); xpu_free(d_b); xpu_free(d_c); xpu_free(d_workspace);
infiniopDestroyTensorDescriptor(a_desc);
infiniopDestroyTensorDescriptor(b_desc);
infiniopDestroyTensorDescriptor(c_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 基础设施依赖
本模块构建在 Infiniop 的通用 elementwise 框架之上，复用了以下基础设施：

- **`ELEMENTWISE_DESCRIPTOR` 宏**（`infiniop/elementwise/elementwise.h`）：
  - 自动生成 `Descriptor` 类的完整定义
  - 封装 `_dtype`, `_info`, `_device_info`, `_workspace_size` 成员变量
  - 声明 `create` 和 `calculate` 静态方法

- **`op::elementwise::ElementwiseInfo`**（`infiniop/elementwise/elementwise.h`）：
  - 存储输入/输出张量的元数据：形状（shape）、步长（strides）、连续性（contiguous）、广播标志（broadcasted）
  - 提供紧凑的内存布局：`output_shape + output_strides + input_shapes[] + input_strides[] + input_contiguous[] + input_broadcasted[]`
  - 通过 `create` 静态方法从张量描述符构造，自动计算内存布局

- **`op::elementwise::kunlun::DeviceImpl`**（`infiniop/elementwise/kunlun/elementwise_kunlun.h`）：
  - 封装昆仑设备端的 kernel 启动逻辑
  - 实现 `calculate<BLOCK_SIZE, Op, Tdata>` 模板方法
  - 管理设备句柄（`device::kunlun::Handle::Internal`）

### Kernel 启动流程
当调用 `Descriptor::calculate` 时，执行以下步骤：

1. **类型分发**：根据 `_dtype` 选择正确的模板特化（half/bfloat16_t/float）

2. **元数据传输**（`DeviceImpl::Opaque::infoToDevice`）：
   - 在设备端工作空间中布局元数据：
     - `workspace[0:input_arr_size]`: 输入指针数组
     - `workspace[input_arr_size:]`: ElementwiseInfo 元数据
   - 使用 `xpu_memcpy_async` 异步复制元数据到设备
   - 计算各元数据段在设备工作空间中的偏移量指针

3. **Kernel 启动**（`elementwiseKernel<N, Op, Tdata>`）：
   - 启动配置：`<<<BLOCK_SIZE, 64, stream>>>`
     - `BLOCK_SIZE = 8`: 集群/线程块数量
     - `64`: 每个集群的线程数（固定）
   - 线程映射：
     - `cid = core_id()`: 集群内的核心 ID（0~63）
     - `ncores = core_num()`: 集群内的核心总数
     - `thread_id = ncores * cluster_id() + cid`: 全局线程 ID
     - `nthreads = ncores * cluster_num()`: 总线程数

4. **设备端计算**（在 `elementwiseKernel` 内部）：
   - **本地内存加载**：
     - 使用 `GM2LM_ASYNC` 将元数据从全局内存加载到本地内存
     - 元数据包括：`input_contiguous[N]`, `input_broadcasted[N]`, `input_shapes[N*ndim]`, `input_strides[N*ndim]`, `output_shape[ndim]`, `output_strides[ndim]`, `typed_inputs_ptr[N]`
     - 调用 `mfence()` 确保加载完成
   - **元素级循环**：
     - 每个线程处理 `len_per_loop = min(64, roundup_div(output_size, nthreads))` 个元素
     - 循环步长为 `nthreads * len_per_loop`
   - **索引计算**：
     - 输出索引：`getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides)`
       - 如果连续：直接使用 `idx`
       - 否则：调用 `indexToOffset` 根据形状和步长计算线性索引
     - 输入索引：`InputIndexer` 函数对象为每个输入计算 `indexer(i)`
   - **算子执行**（`launchOp<N, Op, Tdata>`）：
     - 使用 `GM2LM_ASYNC` 从全局内存异步加载 N 个输入到本地内存 `inputs_buf[N]`
     - 调用 `Op{}(inputs_buf)` 执行乘法运算
     - 使用 `LM2GM_ASYNC` 将结果异步写回全局内存
     - 每次内存操作后调用 `mfence()` 确保内存序

5. **集群同步**：
   - 调用 `sync_cluster()` 确保集群内所有核心完成计算

### 内存管理
- **工作空间分配**：
  - 大小 = `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
  - `getMetaMemSize()` 包括所有张量的形状、步长、连续性标志、广播标志
  - 额外 `input_size * sizeof(void*)` 用于存储输入指针数组
  - 由调用方在设备端分配，传入 `calculate` 方法

- **内存层次**：
  - 全局内存（GM）：存储输入/输出张量数据和元数据
  - 本地内存（LM）：每个线程私有的缓存，用于存储当前元素的输入和元数据
  - 共享内存（SM）：未在此算子中使用

- **内存传输优化**：
  - 使用 `GM2LM_ASYNC` 和 `LM2GM_ASYNC` 异步传输
  - 在 `elementwiseKernel` 启动前，所有元数据通过 `xpu_memcpy_async` 异步复制到设备
  - Kernel 内部使用 `mfence()` 确保异步传输完成后再访问数据

### 并发策略
- **线程级并行**：
  - 采用细粒度元素级并行：每个线程处理一个或多个输出元素
  - 线程映射采用 2D 层次结构：`thread_id = ncores * cluster_id() + cid`
  - 负载均衡：通过 `roundup_div` 和循环步长确保均匀分配

- **SIMD 向量化**：
  - 昆仑 XPU 支持 512-bit SIMD 指令（通过 `xpu/kernel/xtdk_simd.h`）
  - `float32x16_t`: 16 个 float32 的 SIMD 向量（512-bit）
  - `float16x32_t`: 32 个 float16 的 SIMD 向量（512-bit）
  - 当前实现未显式使用向量化，但编译器可能自动向量化 `MulOp` 的简单乘法

- **同步机制**：
  - `mfence()`: 本地内存栅栏，确保本地内存操作完成
  - `sync_cluster()`: 集群同步，确保集群内所有核心完成计算
  - 无显式原子操作：乘法是独立的逐元素操作，无需跨线程同步

### 性能优化
- **连续性检测**：
  - `ElementwiseInfo` 为每个张量存储 `isContiguous()` 标志
  - 连续张量：直接使用线性索引，避免 `indexToOffset` 的除法和模运算
  - 非连续张量：通过 `indexToOffset` 根据步长计算实际偏移量

- **广播支持**：
  - 虽然 mul 的 `CHECK_SAME_SHAPE` 宏要求形状完全相同，但 elementwise 框架支持广播
  - 广播信息存储在 `input_broadcasted[]` 数组中
  - `InputIndexer` 自动处理广播逻辑（通过 `indexToOffset`）

- **批量处理**：
  - 每个线程在本地循环中处理 `len_per_loop` 个元素（最多 64 个）
  - 减少全局内存访问次数，提高数据局部性

- **模板特化优化**：
  - BF16 使用 float 计算精度：避免 `bfloat16_t * bfloat16_t` 的精度损失
  - 类型安全：编译期为每种数据类型生成专用 kernel，避免运行时分支

### 错误处理
- **参数验证**：
  - `CHECK_DTYPE` 宏：拒绝非 FP16/F32/BF16 的数据类型
  - `CHECK_SAME_SHAPE` 宏：要求输入和输出形状完全一致
  - 工作空间大小检查：返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`

- **错误传播**：
  - `ElementwiseInfo::create` 返回 `Result<ElementwiseInfo>`，可能携带错误状态
  - `CHECK_RESULT` 宏：检查 Result 并在出错时提前返回
  - `CHECK_KUNLUN` 宏：检查昆仑 XPU API 调用状态

- **边界情况**：
  - 零大小张量：`launchElementwiseKernel` 检测 `output_size == 0` 并直接返回成功
  - 空指针：在 `ElementwiseInfo::create` 中检查 `output_desc` 和 `input_descs`

### 设备依赖
- **昆仑 XPU 特定 API**：
  - `xpu/runtime.h`: 昆仑 XPU 运行时接口
  - `xpu/kernel/xtdk_*.h`: 昆仑内核开发工具包（数学、SIMD、原子操作等）
  - `__global_ptr__`, `__local_ptr__`, `__shared_ptr__`: 昆仑内存空间修饰符
  - `GM2LM_ASYNC`, `LM2GM_ASYNC`: 异步内存传输指令
  - `core_id()`, `cluster_id()`, `core_num()`, `cluster_num()`: 线程/集群信息查询

- **设备端类型定义**（`kunlun_kernel_common.h`）：
  - `_size_t`: 64 位结构体，包含 32 位 `value` 和 32 位 `padding`
  - `_ptrdiff_t`: 64 位结构体，包含 32 位 `value` 和 32 位 `padding`
  - 用于确保数据对齐，优化 DATACOPY 指令性能

### 设计模式
- **策略模式**：`MulOp` 作为策略对象，通过 `operator()` 定义乘法语义
- **模板方法模式**：`elementwise_kunlun.h` 定义算法骨架（元数据传输、kernel 启动），`MulOp` 提供具体计算逻辑
- **工厂模式**：`Descriptor::create` 作为静态工厂方法，封装复杂的对象构造逻辑
- **Pimpl 惯用法**：`DeviceImpl` 使用 `Opaque` 指针隐藏实现细节（`device::kunlun::Handle::Internal`）
- **RAII**：使用 `std::unique_ptr` 和 `std::shared_ptr` 管理资源生命周期

### 编译与链接
- **文件扩展名**：`.xpu` 文件是昆仑 XPU 的源文件，使用 XPU 编译器（类似 NVIDIA 的 `.cu` 文件）
- **头文件包含路径**：
  - `../../../elementwise/kunlun/elementwise_kunlun.h`: 相对于 `ops/mul/kunlun/` 的路径
  - `../../../utils.h`: 通用工具宏
  - `../../devices/kunlun/kunlun_*.h`: 昆仑设备句柄和通用内核工具
- **宏依赖**：
  - `ELEMENTWISE_DESCRIPTOR`, `CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR`: 来自 elementwise 框架
  - `CHECK_DTYPE`, `CHECK_SAME_SHAPE`, `CHECK_RESULT`, `CHECK_KUNLUN`, `CHECK_STATUS`: 错误检查宏
