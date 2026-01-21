# Moore Backend Add Operation Implementation Documentation

## 概述

本模块实现了 Moore（Moore Threads GPU）硬件后端的张量加法运算操作。它是 Infini 框架中逐元素运算体系的一部分，通过复用通用的逐元素运算基础设施，为 Moore GPU 提供高性能的向量加法内核实现。

该模块支持 FP16、BF16、FP32、FP64、INT32、INT64 六种数据类型，利用 Moore 平台的原生指令进行优化计算。

---

## 1. 模块结构

- **`add_moore.h`** (186 字节)：模块公共 API 接口定义，通过宏声明生成 Descriptor 类框架
- **`add_moore_kernel.h`** (1.4 KB)：核心加法操作符定义，针对不同数据类型实现设备端加法逻辑
- **`add_moore.mu`** (2.3 KB)：Descriptor 类实现，包含构造器创建函数和计算调度函数

### 依赖关系图

```
add_moore.h
    └── elementwise_moore_api.h (逐元素运算 Moore 后端 API)
            └── elementwise.h (通用逐元素运算基础)

add_moore.mu
    ├── add_moore.h (自身 API 定义)
    ├── elementwise_moore.h (逐元素运算 Moore 内核实现)
    └── add_moore_kernel.h (加法操作符定义)

add_moore_kernel.h
    └── 无直接依赖（纯操作符定义）
```

---

## 2. 核心类与数据结构

### 2.1 `op::add::moore::Descriptor` 类

**位置**：通过 `ELEMENTWISE_DESCRIPTOR(add, moore)` 宏在 `add_moore.h` 中自动生成

**继承关系**：继承自 `InfiniopDescriptor` 基类

**主要职责**：封装加法操作的元数据、设备实现和执行接口

**核心成员变量**：
- `infiniDtype_t _dtype`：操作数据类型（FP16/FP32/BF16/FP64/INT32/INT64）
- `op::elementwise::ElementwiseInfo _info`：张量形状、步幅、广播等元数据
- `std::unique_ptr<op::elementwise::moore::DeviceImpl> _device_info`：Moore 设备端实现指针
- `size_t _workspace_size`：所需显存工作空间大小

**生命周期管理**：
- **构造**：通过静态工厂方法 `create()` 创建，内部调用 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏
- **析构**：默认析构函数（`~Descriptor() = default;`）
- **所有权**：调用方拥有返回的 Descriptor 指针，负责释放

### 2.2 `op::add::moore::AddOp` 结构体

**位置**：`add_moore_kernel.h` 第 13-35 行

**类型**：函数对象（Functor）

**模板参数**：
- `typename T`：数据类型（half2, half, cuda_bfloat16, float, 或其他算术类型）

**核心方法**：
```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &a, const T &b) const
```

**实现策略**（基于 `if constexpr` 编译期分支）：

| 数据类型 | 实现方式 | 指令/方法 |
|---------|---------|----------|
| `half2` (FP16x2 向量) | 直接调用原生 FP16 向量加法 | `__hadd2(a, b)` |
| `half` (FP16 标量) | 原生 FP16 标量加法 | `__hadd(a, b)` |
| `cuda_bfloat16` (BF16) | 转换到 FP32 计算，再转回 BF16 | `__bfloat162float` → `+` → `__float2bfloat16_rn` |
| `float` (FP32) | Moore 平台兼容的舍入模式加法 | `__fadd_rn(a, b)` |
| 其他类型（int32_t, int64_t, double 等） | C++ 原生加法运算符 | `a + b` |

**设计考量**：
- BF16 采用间接路径是因为 Moore 平台上 `__hadd` 返回 `int` 类型，导致到 `__mt_bfloat16` 的转换歧义
- FP32 使用 `__fadd_rn`（round-to-nearest）而非 `__fadd_rd`（round-down）以兼容 Moore 平台规范

### 2.3 `op::elementwise::ElementwiseInfo` 结构体

**位置**：定义于 `elementwise.h`，被本模块通过继承/组合使用

**内存布局**（紧凑打包于单个 `std::vector<size_t>` 中）：
```
[output_shape (ndim * size_t)]
[output_strides (ndim * ptrdiff_t)]
[input_shapes (input_size * ndim * size_t)]
[input_strides (input_size * ndim * ptrdiff_t)]
[input_contiguous (input_size * bool)]
[input_broadcasted (input_size * bool)]
```

**关键方法**：
- `getMetaMemSize()`：返回元数据总字节数
- `getOutputSize()`：返回输出张量元素总数
- `getInputSize()`：返回输入张量数量
- `isOutputContiguous()`：输出是否内存连续
- `getInputShape(index)`, `getInputStrides(index)`：获取指定输入的形状/步幅
- `getInputContiguous()`, `getInputBroadcasted()`：获取连续性和广播标志数组

### 2.4 `op::elementwise::moore::DeviceImpl` 类

**位置**：定义于 `elementwise_moore_api.h` 和 `elementwise_moore.h`

**Pimpl 模式**：通过 `struct Opaque` 隐藏实现细节

**核心模板方法**：
```cpp
template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t calculate(
    const ElementwiseInfo &info,
    void *workspace, void *output,
    const std::vector<const void *> &inputs,
    void *stream, Args &&...args);
```

**实现细节**（详见 `elementwise_moore.h`）：
1. **元数据传输**：通过 `infoToDevice<N>()` 将形状、步幅等元数据异步复制到设备显存
2. **内核启动配置**：
   - `blockDims.x = min(BLOCK_SIZE, maxThreadsPerBlock)`，默认 BLOCK_SIZE=256
   - `gridDims.x = min(ceil_div(output_size, blockDims.x), gridSizeX)`
3. **分段执行**：使用 `for` 循环处理超大规模张量（`i += step`），每次启动一个网格

---

## 3. API 接口

### 3.1 Descriptor 创建接口

```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // [in] Moore 设备句柄
    Descriptor **desc_ptr,                  // [out] 输出的 Descriptor 指针
    infiniopTensorDescriptor_t out_desc,    // [in] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量 {A, B}
);
```

**返回值**：
- `INFINI_STATUS_SUCCESS`：创建成功
- `INFINI_STATUS_BAD_TENSOR_DTYPE`：不支持的数据类型
- `INFINI_STATUS_BAD_TENSOR_STRIDES`：形状不匹配或步幅非法

**前置条件检查**（第 26-28 行）：
```cpp
CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
            INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);
```

**执行流程**：
1. 类型检查与形状校验
2. 调用 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏：
   - 创建 `ElementwiseInfo` 对象（包含形状/步幅/广播信息）
   - 创建 `DeviceImpl` 对象
   - 计算工作空间大小 = `metaMemSize + inputSize * sizeof(void*)`
   - 构造并返回 `Descriptor` 实例

### 3.2 计算执行接口

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,                         // [in] 设备显存工作空间指针
    size_t workspace_size,                   // [in] 工作空间大小（字节）
    void *output,                            // [out] 输出张量设备指针
    std::vector<const void *> inputs,        // [in] 输入张量设备指针数组 {A, B}
    void *stream                             // [in] MUSA 流句柄
) const;
```

**返回值**：
- `INFINI_STATUS_SUCCESS`：计算成功
- `INFINI_STATUS_INSUFFICIENT_WORKSPACE`：工作空间不足
- `INFINI_STATUS_BAD_TENSOR_DTYPE`：未知数据类型

**实现逻辑**（第 47-62 行，类型分发 switch）：
```cpp
switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::AddOp, half>(
            _info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::AddOp, cuda_bfloat16>(
            _info, workspace, output, inputs, stream);
    // ... 其他类型分支
}
```

**关键模板参数**：
- `256`：CUDA/MUSA 线程块大小（每 block 256 线程）
- `moore::AddOp`：加法操作符
- `half` / `cuda_bfloat16` / `float` 等：具体数据类型

---

## 4. 使用示例

### 4.1 基本用法

```cpp
#include "infiniop/ops/add/moore/add_moore.h"

// 1. 初始化 Moore 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(INFINI_DEVICE_MOORE, 0, &handle);

// 2. 创建张量描述符（假设形状为 [1024, 1024]）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};  // C 风格连续内存

infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 2, shape, strides, &a_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 2, shape, strides, &b_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 2, shape, strides, &c_desc);

// 3. 创建加法操作描述符
op::add::moore::Descriptor* add_desc;
std::vector<infiniopTensorDescriptor_t> inputs = {a_desc, b_desc};
auto status = op::add::moore::Descriptor::create(handle, &add_desc, c_desc, inputs);

// 4. 分配设备显存
size_t workspace_size = add_desc->workspaceSize();
void* d_workspace;
void* d_a, *d_b, *d_c;
musaMalloc(&d_workspace, workspace_size);
musaMalloc(&d_a, 1024 * 1024 * sizeof(half));
musaMalloc(&d_b, 1024 * 1024 * sizeof(half));
musaMalloc(&d_c, 1024 * 1024 * sizeof(half));

// 5. 准备输入数据（省略主机到设备复制）
// musaMemcpyAsync(d_a, h_a, ..., musaMemcpyHostToDevice, stream);

// 6. 获取流并执行计算
musaStream_t stream;
musaStreamCreate(&stream);

std::vector<const void*> input_ptrs = {d_a, d_b};
status = add_desc->calculate(d_workspace, workspace_size, d_c, input_ptrs, stream);

// 7. 同步并获取结果
musaStreamSynchronize(stream);
// musaMemcpyAsync(h_c, d_c, ..., musaMemcpyDeviceToHost, stream);

// 8. 清理资源
musaFree(d_workspace); musaFree(d_a); musaFree(d_b); musaFree(d_c);
musaStreamDestroy(stream);
delete add_desc;
infiniopDestroyHandle(handle);
```

### 4.2 广播场景示例

```cpp
// 场景：将形状 [1024] 的向量加到形状 [1024, 1024] 的矩阵上
int64_t matrix_shape[] = {1024, 1024};
int64_t matrix_strides[] = {1024, 1};
int64_t vector_shape[] = {1024};
int64_t vector_strides[] = {1};

infiniopTensorDescriptor_t matrix_desc, vector_desc, output_desc;
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, 2,
                               matrix_shape, matrix_strides, &matrix_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, 1,
                               vector_shape, vector_strides, &vector_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, 2,
                               matrix_shape, matrix_strides, &output_desc);

op::add::moore::Descriptor* add_desc;
std::vector<infiniopTensorDescriptor_t> inputs = {matrix_desc, vector_desc};
op::add::moore::Descriptor::create(handle, &add_desc, output_desc, inputs);

// 计算时，框架会自动处理维度对齐和广播逻辑
add_desc->calculate(d_workspace, workspace_size, d_output,
                    {d_matrix, d_vector}, stream);
```

---

## 5. 实现细节

### 5.1 内存管理

**工作空间布局**（总大小 = `metaMemSize + inputSize * sizeof(void*)`）：
```
+-------------------+  <- workspace 起始地址
| 输入指针数组      |  sizeof(void*) * input_size 字节
| (input pointers)  |
+-------------------+
| 输出形状          |  sizeof(size_t) * ndim 字节
+-------------------+
| 输出步幅          |  sizeof(ptrdiff_t) * ndim 字节
+-------------------+
| 所有输入形状      |  sizeof(size_t) * input_size * ndim 字节
+-------------------+
| 所有输入步幅      |  sizeof(ptrdiff_t) * input_size * ndim 字节
+-------------------+
| 输入连续标志      |  sizeof(bool) * input_size 字节
+-------------------+
| 输入广播标志      |  sizeof(bool) * input_size 字节
+-------------------+
```

**数据流向**：
1. 主机端构建 `ElementwiseInfo`（内存中的 `std::vector<size_t>`）
2. 调用 `calculate` 时，通过 `musaMemcpyAsync` 将元数据复制到设备工作空间
3. 设备端内核从全局显存读取元数据，计算索引后访问实际输入/输出数据

**所有权模型**：
- `Descriptor` 拥有 `DeviceImpl` 的 `unique_ptr`
- `DeviceImpl` 通过 Pimpl 模式拥有 `Opaque` 对象
- `Opaque` 持有 `shared_ptr<device::moore::Handle::Internal>`（共享设备上下文）

### 5.2 并发执行

**流式执行**：
- 所有 MUSA 操作（内存复制、内核启动）都绑定到用户提供的 `musaStream_t`
- 支持多流并发：同一 Descriptor 可在不同流上多次调用 `calculate`
- 无内部锁：假设调用方保证流的线程安全性

**内核启动配置**（`elementwise_moore.h` 第 204-216 行）：
```cpp
dim3 blockDims(std::min(BLOCK_SIZE, internal->maxThreadsPerBlock()));
dim3 gridDims(std::min(uint32_t(CEIL_DIV(output_size, blockDims.x)),
                       internal->gridSizeX()));
size_t step = gridDims.x * blockDims.x;

for (size_t i = 0; i < output_size; i += step) {
    kernel_func<<<gridDims, blockDims, 0, stream>>>(..., i, ...);
}
```

**性能优化点**：
- 使用 256 线程/块（平衡寄存器使用和占用率）
- 分段启动内核处理超过 `gridSizeX * blockDim.x` 的张量
- 编译器内联（`__forceinline__`）减少函数调用开销

### 5.3 错误处理

**错误传播机制**（基于 `infiniStatus_t` 枚举）：

| 错误码 | 触发场景 | 处理方式 |
|--------|---------|---------|
| `INFINI_STATUS_BAD_PARAM` | 空指针传入 | 立即返回错误 |
| `INFINI_STATUS_BAD_TENSOR_DTYPE` | 不支持的 dtype | switch default 分支返回 |
| `INFINI_STATUS_BAD_TENSOR_STRIDES` | 形状不匹配 | `CHECK_SAME_SHAPE` 宏检查 |
| `INFINI_STATUS_INSUFFICIENT_WORKSPACE` | workspace_size 过小 | `calculate` 前置检查 |

**宏辅助检查**（定义于 `utils.h`）：
```cpp
CHECK_DTYPE(dtype, ...)      // 类型白名单检查
CHECK_SAME_SHAPE(...)        // 形状一致性断言
CHECK_RESULT(expr)           // Result<T> 类型解包
CHECK_MOORE(expr)            // MUSA API 调用错误检查
```

**无异常设计**：所有错误通过返回值传播，不抛出 C++ 异常（符合 CUDA/MUSA 编程规范）

### 5.4 性能特性

**时间复杂度**：
- 理论上为 O(N)，N 为输出张量元素总数
- 每个线程处理一个元素，完全并行

**空间复杂度**：
- 额外显存 = O(ndim * input_size)（元数据存储）
- 工作空间 = O(1) 对输入规模（仅元数据，无数据复制）

**硬件加速**：
- `half2` 类型利用 FP16 向量指令（2 倍吞吐量）
- BF16 通过 FP32 路径规避硬件限制
- FP32 使用舍入到最近模式（`__fadd_rn`）保证数值精度

### 5.5 设备端内核实现

**内核签名**（`elementwise_moore.h` 第 42-70 行）：
```cpp
template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_MOORE_KERNEL elementwiseKernel(
    size_t output_size, size_t ndim, bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tdata *output, const void *const *inputs,
    size_t offset, Args... args);
```

**索引计算逻辑**：
1. **线性索引**：`idx = blockIdx.x * blockDim.x + threadIdx.x + offset`
2. **边界检查**：`if (idx < output_size)`（防止越界）
3. **输出索引**：
   - 连续内存：`out_idx = idx`
   - 非连续内存：`out_idx = indexToOffset(idx, ndim, shape, strides)`
4. **输入索引**：通过 `InputIndexer` 函数对象按需计算每个输入的偏移

**操作符调用**（第 64-68 行）：
```cpp
unpackInputsAndApply(
    [&](auto... Is) {
        output[out_idx] = Op{}(typed_inputs[Is.value][indexer(Is.value)]...,
                               std::forward<Args>(args)...);
    },
    std::make_index_sequence<N>{});
```

使用 C++17 折叠表达式和编译期索引序列展开，实现通用的 N 参数操作符调用。

### 5.6 设计模式总结

**模式 1：宏生成代码模式（Code Generation via Macro）**
- `ELEMENTWISE_DESCRIPTOR(add, moore)` 宏自动生成完整的 Descriptor 类
- 避免为每个操作符重复编写样板代码

**模式 2：策略模式（Strategy Pattern）**
- `AddOp` 作为可插拔的策略对象
- `DeviceImpl::calculate<Op, T>()` 通过模板参数接受任意操作符

**模式 3：Pimpl 惯用法（Pointer to Implementation）**
- `DeviceImpl` 通过 `Opaque` 结构体隐藏实现细节
- 减少编译依赖和头文件暴露

**模式 4：类型擦除（Type Erasure）**
- 公共 API 使用 `void *` 和 `infiniDtype_t` 掩盖具体类型
- 内部通过模板和 switch 恢复类型信息进行优化

**模式 5：RAII 资源管理**
- `ElementwiseInfo` 使用移动语义管理元数据内存
- `Descriptor` 使用 `unique_ptr` 管理 `DeviceImpl` 生命周期

---

## 6. 依赖与兼容性

### 6.1 外部依赖

| 依赖组件 | 版本要求 | 用途 |
|---------|---------|------|
| MUSA SDK | Moore Threads 平台 | 设备端编译、内核启动 |
| CUDA 兼容层 | 用于类型定义 | `half`, `cuda_bfloat16`, `half2` |
| Infini 基础设施 | 同仓库 | 张量描述符、设备句柄、工具宏 |

### 6.2 平台特性

**Moore vs CUDA 差异**：
- 使用 `musaStream_t` 替代 `cudaStream_t`
- 使用 `INFINIOP_MOORE_KERNEL` 替代 `__global__`
- BF16 计算路径需特殊处理（`__bfloat162float` 转换）

**向后兼容性**：
- 代码结构保持与 CUDA 实现一致（命名空间 `op::add::moore`）
- 通过替换设备后端支持多平台编译

---

## 7. 扩展指南

### 7.1 添加新的数据类型支持

在 `add_moore_kernel.h` 的 `AddOp::operator()` 中添加新分支：
```cpp
else if constexpr (std::is_same_v<T, new_type>) {
    // 实现新类型的加法逻辑
}
```

在 `add_moore.mu` 的 `calculate` 方法中添加 case：
```cpp
case INFINI_DTYPE_NEW_TYPE:
    return _device_info->calculate<256, moore::AddOp, new_type>(
        _info, workspace, output, inputs, stream);
```

### 7.2 添加新的逐元素运算

1. 创建新目录 `infiniop/ops/new_op/moore/`
2. 参考 `add` 操作的文件结构创建对应文件
3. 在 `new_op_kernel.h` 中定义操作符（确保有 `num_inputs` 静态成员）
4. 使用宏 `ELEMENTWISE_DESCRIPTOR(new_op, moore)` 生成类框架
5. 实现 `create` 和 `calculate` 方法

---

## 8. 已知限制与未来改进

**当前限制**：
1. 不支持混合精度运算（所有输入必须同 dtype）
2. 仅支持二元操作（`num_inputs = 2`）
3. 大张量需要多次内核启动（虽已优化但仍存在循环开销）

**潜在优化方向**：
1. 引入向量化内存加载（如 `float4`, `half4`）提升带宽利用率
2. 针对连续内存张量使用专用内核（避免索引计算）
3. 支持融合操作（如 Add + ReLU 单次内核完成）

---

## 附录：关键文件路径

- 主实现：`/home/qy/src/Infini/InfiniCore/src/infiniop/ops/add/moore/add_moore.mu`
- 内核定义：`/home/qy/src/Infini/InfiniCore/src/infiniop/ops/add/moore/add_moore_kernel.h`
- API 接口：`/home/qy/src/Infini/InfiniCore/src/infiniop/ops/add/moore/add_moore.h`
- 逐元素基础设施：`/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
- Moore 设备实现：`/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/moore/elementwise_moore.h`
- Moore API 层：`/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/moore/elementwise_moore_api.h`
