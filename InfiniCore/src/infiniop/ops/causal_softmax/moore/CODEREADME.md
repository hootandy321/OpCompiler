# Moore后端Causal Softmax算子实现文档

## 概述

本模块实现了**Moore GPU**（摩尔线程GPU，MUSA架构）上的因果掩码Softmax操作。该算子主要用于Transformer模型中的自注意力机制，在计算注意力分数时应用因果掩码（即位置i只能关注位置≤i的内容），确保自回归生成的正确性。

核心算法采用三阶段并行策略：先找到每行最大值（数值稳定），然后计算指数并应用因果掩码，最后求和归一化。针对MUSA平台的特殊性和硬件架构差异（如512/1024线程块限制），实现了专门的类型转换和兼容性处理。

---

## 1. 模块结构

```
moore/
├── causal_softmax_kernel.h         # CUDA/MUSA设备端核函数实现
├── causal_softmax_moore.h          # Moore后端Descriptor声明（宏定义）
└── causal_softmax_moore.mu         # Moore后端完整实现（主机端+内核启动）
```

**文件职责**：

- **`causal_softmax_kernel.h`**：实现设备端的核心计算逻辑，包含模板化的`causalSoftmaxKernel`函数，执行数值稳定的softmax+因果掩码操作。

- **`causal_softmax_moore.h`**：通过`DESCRIPTOR(moore)`宏展开生成`op::causal_softmax::moore::Descriptor`类声明，继承自基类`InfiniopDescriptor`。

- **`causal_softmax_moore.mu`**：实现`Descriptor`的主机端方法（构造、create、calculate），包含内核启动逻辑和设备能力查询（最大线程块大小）。

---

## 2. 核心数据结构

### `CausalSoftmaxInfo`
**位置**：`../info.h`（共享定义）

存储输入输出张量的元信息，由工厂方法`create`验证并构造。

**成员变量**：
- `infiniDtype_t dtype`：数据类型（F16/BF16/F32）
- `size_t batch_size`：批次大小
- `size_t seq_len`：序列长度（行数，即height）
- `size_t total_seq_len`：总序列长度（列数，即width）
- `ptrdiff_t y_stride_b/i/j`：输出张量在批次、行、列维度的步长
- `ptrdiff_t x_stride_b/i/j`：输入张量在批次、行、列维度的步长

**约束条件**（在`create`中验证）：
- 输入输出数据类型必须一致
- 支持2D `[seq_len, total_seq_len]` 或 3D `[batch_size, seq_len, total_seq_len]`
- 必须满足 `total_seq_len >= seq_len`（因果掩码的语义要求）

**生命周期**：值对象，在`Descriptor::create`中构造后存储为成员变量`_info`。

---

### `op::causal_softmax::moore::Descriptor::Opaque`
**位置**：`causal_softmax_moore.mu`

```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};
```

**职责**：封装Moore设备句柄的内部实现（`Handle::Internal`），使用`shared_ptr`管理生命周期，确保与外部`infiniopHandle_t`的同步。

**关键访问能力**：
- `internal->maxThreadsPerBlock()`：查询设备最大线程块大小，用于选择内核配置（512或1024）。

---

## 3. 核心类与函数

### 3.1 设备端核函数：`causalSoftmaxKernel`

**签名**：
```cpp
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void causalSoftmaxKernel(
    Tdata *y_, const Tdata *x_,
    size_t batch, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_h);
```

**参数**：
- `BLOCK_SIZE`：线程块大小（编译时常量，512或1024）
- `Tdata`：数据类型（`half`, `__mt_bfloat16`, `float`）
- `Tcompute`：计算类型（通常为`float`，保证数值精度）
- `y_`：输出张量基地址
- `x_`：输入张量基地址
- `batch/height/width`：张量形状（批次、行、列）
- `*_stride_*`：各维度步长

**线程映射**：
- **Grid**：`gridDim.x = height`（行数），`gridDim.y = batch`（批次）
- **Block**：`threadIdx.x ∈ [0, BLOCK_SIZE-1]`（列维度并行）

**算法流程**（4阶段同步协作）：

#### 阶段1：寻找最大值（数值稳定）
```cpp
__shared__ Tdata max_;
Tdata max_0 = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(
    x, width - height + 1 + blockIdx.x  // 因果掩码区域的有效长度
);
if (threadIdx.x == 0) max_ = max_0;
__syncthreads();
```

使用CUB的`BlockReduce`进行归约，仅`threadIdx.x == 0`获得正确结果。归约长度动态计算为`width - height + blockIdx.x + 1`，对应每行的有效元素数量（因果掩码下的右下三角形）。

#### 阶段2：指数运算+因果掩码
```cpp
for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
    // 因果掩码条件：width + blockIdx.x >= threadIdx.x + height
    // 等价于：col >= blockIdx.x（当前行号）
    if (width + blockIdx.x >= threadIdx.x + height) {
        // 针对半精度类型的MUSA兼容性处理
        float val = static_cast<float>(x[col]) - static_cast<float>(max_);
        y[col] = static_cast<Tdata>(expf(val));
    } else {
        y[col] = Tdata(0.0f);  // 被掩码的位置置零
    }
}
__syncthreads();
```

**因果掩码逻辑**：
- 条件`width + blockIdx.x >= threadIdx.x + height`等价于`col >= row_id`（其中`row_id = blockIdx.x`）
- 位置(i, j)合法当且仅当`j >= i`（行i只能看列j≥i）
- 不合法位置输出显式置为`0.0f`（MUSA的`__mt_bfloat16`需要显式浮点字面量避免构造歧义）

#### 阶段3：求和
```cpp
__shared__ Tcompute sum_;
Tcompute sum_0 = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(y, width);
if (threadIdx.x == 0) sum_ = sum_0;
__syncthreads();
```

对指数运算后的结果进行全行求和，计算类型为`Tcompute`（通常`float`），避免半精度溢出。

#### 阶段4：归一化
```cpp
for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
    // MUSA bfloat16不支持 `/=` 操作符重载，显式转换为float
    y[col] = static_cast<Tdata>(static_cast<float>(y[col]) / static_cast<float>(sum_));
}
```

每个元素除以总和，完成softmax归一化。MUSA平台限制要求显式类型转换。

**复杂度**：
- 时间：`O(width / BLOCK_SIZE)` 每线程，加上`O(log BLOCK_SIZE)`的归约开销
- 空间：每个Block使用2个共享内存变量（`max_`, `sum_`），共`sizeof(Tdata)+sizeof(Tcompute)`字节

---

### 3.2 主机端接口：`Descriptor::create`

**签名**：
```cpp
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc);
```

**职责**：工厂方法，验证张量描述符并构造Descriptor实例。

**执行流程**：
1. 调用`CausalSoftmaxInfo::create(y_desc, x_desc)`验证形状和数据类型
2. 检查返回的`Result`，失败时传播错误码（如`INFINI_STATUS_BAD_TENSOR_DTYPE`）
3. 提取Moore设备句柄：`reinterpret_cast<device::moore::Handle *>(handle)->internal()`
4. 构造`Descriptor`实例并赋值给`*desc_ptr`

**错误处理**：
- 数据类型不匹配：`INFINI_STATUS_BAD_TENSOR_DTYPE`
- 形状非法（如`total_seq_len < seq_len`）：`INFINI_STATUS_BAD_TENSOR_SHAPE`

---

### 3.3 计算接口：`Descriptor::calculate`

**签名**：
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream_) const;
```

**职责**：启动内核执行实际计算。

**执行流程**：
1. 转换流：`musaStream_t stream = (musaStream_t)stream_`
2. 查询设备能力：`_opaque->internal->maxThreadsPerBlock()`
3. 根据返回值选择模板实例：
   - `1024` → 调用`launchKernel<1024>`
   - `512` → 调用`launchKernel<512>`
   - 其他 → 返回`INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
4. `launchKernel`内部根据`_info.dtype`分发到不同特化版本（F16/BF16/F32）

**内核启动配置**：
```cpp
dim3 grid(uint32_t(seq_len), uint32_t(batch_size), 1);
causalSoftmax<BLOCK_SIZE, Tdata, Tcompute>
    <<<grid, BLOCK_SIZE, 0, stream>>>(/* args */);
```

- Grid大小：`(seq_len, batch_size, 1)`，其中`seq_len = height`
- Block大小：编译时常量`BLOCK_SIZE`（512或1024）
- 动态共享内存：0（仅静态共享内存）

---

## 4. 内核启动分发：`launchKernel`

**签名**：
```cpp
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y, const void *x, infiniDtype_t dtype,
    size_t batch_size, size_t seq_len, size_t total_seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_i,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_i,
    musaStream_t stream);
```

**职责**：根据数据类型分发到对应的内核实例。

**类型映射**：
- `INFINI_DTYPE_F16` → `half`（MUSA的`__half`）
- `INFINI_DTYPE_BF16` → `__mt_bfloat16`（MUSA特有的bfloat16类型）
- `INFINI_DTYPE_F32` → `float`
- 其他 → 返回`INFINI_STATUS_BAD_TENSOR_DTYPE`

---

## 5. 实现细节

### 5.1 MUSA平台兼容性处理

代码中包含大量针对MUSA（摩尔线程CUDA类平台）的特殊处理：

1. **半精度指数运算**（第32-40行）：
   ```cpp
   if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, cuda_bfloat16>) {
       float val = static_cast<float>(x[col]) - static_cast<float>(max_);
       y[col] = static_cast<Tdata>(expf(val));
   }
   ```
   原因：MUSA不支持CUDA的`hexp`原生函数，需显式转为`float`计算`expf`后再转回。

2. **零值初始化**（第55行）：
   ```cpp
   y[col] = Tdata(0.0f);  // 显式使用float字面量
   ```
   原因：`__mt_bfloat16`对整数字面量`0`存在构造歧义（可能从`float`或`double`隐式转换）。

3. **除法操作**（第76行）：
   ```cpp
   y[col] = static_cast<Tdata>(static_cast<float>(y[col]) / static_cast<float>(sum_));
   ```
   原因：MUSA的bfloat16缺少可用的`/=`操作符重载，需先转`float`执行除法。

---

### 5.2 归约操作依赖

依赖通用的CUDA归约库（`op::common_cuda::reduce_op`）：
- `max<BLOCK_SIZE, Tdata>`：使用CUB的`BlockReduce::Reduce`配合`cub::Max`或`cuda::maximum`（CUDA 12.9+）
- `sum<BLOCK_SIZE, Tdata, Tcompute>`：使用CUB的`BlockReduce::Sum`

**重要约束**：归约结果仅在`threadIdx.x == 0`的线程中正确，需要手动广播（通过`__shared__`变量和`__syncthreads()`）。

---

### 5.3 内存管理

- **共享内存**：每个Block使用2个标量共享变量（`max_`, `sum_`），总计约4-8字节（取决于类型）。
- **全局内存**：输入输出张量由调用方管理，内核直接读写。
- **工作空间**：`calculate`方法的`workspace`参数未使用（`workspace_size`为0），所有计算通过寄存器和共享内存完成。

---

### 5.4 线程块大小选择策略

通过查询`Handle::Internal->maxThreadsPerBlock()`动态选择：
- **1024线程**：适用于现代Moore GPU（高吞吐）
- **512线程**：适用于较老或资源受限的GPU
- **不支持的设备**：返回错误码`INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`

这种设计允许代码在多种MUSA架构上运行而无需重新编译。

---

### 5.5 因果掩码的数学原理

对于输入矩阵`X`形状为`[batch, height, width]`，输出`Y`计算为：

```
对于每个位置 (b, i, j)：
  如果 j >= i：
    Y[b,i,j] = exp(X[b,i,j] - max_k(X[b,i,k])) / Σ_{m>=i} exp(X[b,i,m] - max_k(X[b,i,k]))
  否则：
    Y[b,i,j] = 0
```

其中`max_k`是对所有有效列`m >= i`求最大值。这保证了：
1. 下三角区域（j < i）被严格置零
2. 上三角区域（j >= i）执行标准的softmax归一化
3. 数值稳定性（减去最大值避免溢出）

---

## 6. 设计模式

### 6.1 策略模式（Strategy Pattern）
通过模板参数`BLOCK_SIZE`和类型`Tdata/Tcompute`，在编译期生成多个内核变体，运行时根据设备能力选择策略。

### 6.2 工厂模式（Factory Pattern）
`Descriptor::create`作为工厂方法，封装对象构造逻辑和参数验证。

### 6.3 Pimpl模式（Pointer to Implementation）
`Opaque`结构体隐藏`Handle::Internal`的细节，避免头文件暴露MUSA特定类型。

### 6.4 RAII（资源获取即初始化）
`Opaque`使用`std::shared_ptr`管理`Handle::Internal`生命周期，确保与外部句柄同步销毁。

---

## 7. 依赖关系

### 外部依赖

- **MUSA Runtime**：`musa.h`, `musa_runtime_api.h`（内核启动、流管理）
- **MUSA数学库**：`musa_fp16_mtgpu.h`, `musa_bf16.h`（半精度类型定义）
- **CUB**：`cub/block/block_reduce.cuh`（块级归约原语）
- **InfiniOP基础设施**：
  - `../../operator.h`：基类`InfiniopDescriptor`
  - `../../tensor.h`：张量描述符
  - `../../../utils.h`：错误处理宏（`CHECK_STATUS`, `CHECK_RESULT`）
  - `../../../devices/moore/*`：Moore设备抽象层

### 模块间依赖

- **父模块**：`../causal_softmax.h`定义接口宏，`../info.h`定义元信息结构
- **通用工具**：`../../../reduce/cuda/reduce.cuh`提供跨平台归约操作

---

## 8. 性能考量

1. **并行度**：
   - 行间完全并行（每个Block处理一行）
   - 行内按`BLOCK_SIZE`并行（通常512-1024线程）
   - 适合`width >> BLOCK_SIZE`的场景（如长序列模型）

2. **带宽优化**：
   - 每个元素读写3次（读输入、写中间结果、读中间结果、写输出）
   - 可优化：将指数运算和归一化合并到单次遍历（需牺牲部分可读性）

3. **数值精度**：
   - 使用`float`作为计算类型（`Tcompute`）避免半精度溢出
   - 最大值归约保证指数运算稳定性

4. **限制**：
   - 共享内存使用极小，不受共享内存容量限制
   - 寄存器压力主要来自循环展开和局部变量

---

## 9. 使用示例

```cpp
// 假设已有MUSA设备和流
musaStream_t stream;
infiniopHandle_t handle;  // Moore设备句柄

// 定义张量形状：[batch_size=2, seq_len=128, total_seq_len=256]
int64_t shape[] = {2, 128, 256};
int64_t stride[] = {128 * 256, 256, 1};

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensor(handle, &x_desc, INFINI_DTYPE_F16, 3, shape, stride);
infiniopCloneTensor(handle, &y_desc, x_desc);

// 创建算子描述符
op::causal_softmax::moore::Descriptor *softmax_desc;
auto status = op::causal_softmax::moore::Descriptor::create(
    handle, &softmax_desc, y_desc, x_desc);

// 分配GPU内存
half *d_x, *d_y;
musaMalloc(&d_x, 2 * 128 * 256 * sizeof(half));
musaMalloc(&d_y, 2 * 128 * 256 * sizeof(half));

// 执行计算（无需workspace）
status = softmax_desc->calculate(nullptr, 0, d_y, d_x, stream);

// 同步并检查结果
musaStreamSynchronize(stream);

// 清理
musaFree(d_x);
musaFree(d_y);
infiniopDestroyTensor(x_desc);
infiniopDestroyTensor(y_desc);
delete softmax_desc;
```

---

## 10. 总结

本模块针对Moore GPU实现了高性能的因果Softmax算子，核心特点：

1. **算法**：三阶段归约-映射-归一化模式，数值稳定且并行高效
2. **兼容性**：大量MUSA特定处理（类型转换、操作符限制），确保在摩尔线程硬件上正确运行
3. **灵活性**：动态线程块大小选择、多数据类型支持
4. **可维护性**：清晰的模块划分（设备端/主机端分离）、通用归约库复用

该实现是Infini框架中多后端算子生态的一部分，展示了如何将通用算子接口适配到特定GPU平台。
