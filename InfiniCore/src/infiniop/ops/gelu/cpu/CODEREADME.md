# GELU CPU 操作核心实现文档

本模块实现了 GELU (Gaussian Error Linear Unit) 激活函数的 CPU 后端执行，基于统一的逐元素操作 (Elementwise Operation) 框架，提供高性能的多精度数据类型支持。

## 1. 模块结构

- **`gelu_cpu.h`**: 定义 GELU 操作的核心计算算子和宏生成的 Descriptor 类
- **`gelu_cpu.cc`**: 实现 Descriptor 的创建和计算调度逻辑

## 2. 核心数据结构

### `GeluOp` 结构体
- **位置**: `gelu_cpu.h:11-19`
- **功能**: 实现 GELU 激活函数的逐元素计算算子
- **关键成员**:
  - `static constexpr size_t num_inputs = 1`: 标识这是一个单输入操作的算子
- **核心方法**:
  - `template <typename T> T operator()(const T &x) const`: 执行 GELU 计算
    - **算法**: GELU 标准数学公式: `GELU(x) = 0.5 * x * (1 + erf(x / √2))`
    - **实现细节**:
      - 使用 `std::erf` 计算误差函数
      - 使用 `std::sqrt(2.0f)` 计算归一化因子
      - 结果通过 `static_cast<T>` 转换为目标类型
    - **时间复杂度**: O(1) 每元素
    - **数值精度**: 依赖于模板类型 T (bf16_t, fp16_t, float, double)

### `Descriptor` 类 (宏生成)
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(gelu, cpu)` 宏自动生成
- **功能**: 封装 GELU CPU 实现的操作描述符，继承自 `InfiniopDescriptor`
- **关键成员**:
  - `infiniDtype_t _dtype`: 输出张量的数据类型 (BF16, F16, F32, F64)
  - `op::elementwise::ElementwiseInfo _info`: 张量形状、步长、广播等元信息
  - `std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info`: CPU 设备特定的实现对象
  - `size_t _workspace_size`: 工作空间大小 (当前为 0)
- **生命周期**:
  - **创建**: 通过静态方法 `Descriptor::create()` 创建
  - **销毁**: 使用默认析构函数 `~Descriptor()`
  - **所有权**: 调用者持有 Descriptor 指针的所有权

### `ElementwiseInfo` 结构体
- **位置**: `elementwise/elementwise.h:69-203`
- **功能**: 存储逐元素操作的元数据，包括输入/输出张量的形状、步长、连续性等信息
- **内存布局**: 使用单一的 `std::vector<size_t> _meta` 压缩存储所有元数据，通过指针偏移访问不同区域
  - 输出形状 (output_shape): 前 `ndim` 个 `size_t`
  - 输出步长 (output_strides): 接下来的 `ndim` 个 `ptrdiff_t`
  - 输入形状数组 (input_shapes): 接下来的 `input_size * ndim` 个 `size_t`
  - 输入步长数组 (input_strides): 接下来的 `input_size * ndim` 个 `ptrdiff_t`
  - 输入连续性标志 (input_contiguous): 接下来的 `input_size` 个 `bool`
  - 输入广播标志 (input_broadcasted): 最后的 `input_size` 个 `bool`
- **关键方法**:
  - `static ResultType create(...)`: 从张量描述符构造元数据，验证形状兼容性
  - `getOutputSize()`: 返回输出张量的元素总数
  - `isOutputContiguous()`: 判断输出张量是否内存连续
  - `getInputShape(index)`: 获取指定输入的形状指针
  - `getInputStrides(index)`: 获取指定输入的步长指针

## 3. API 接口

### C 风格 API (通过 operator.cc 暴露)
```cpp
// 创建 GELU 描述符
infiniStatus_t infiniopCreateGeluDescriptor(
    infiniopHandle_t handle,                              // [输入] 设备句柄
    infiniopGeluDescriptor_t *desc_ptr,                  // [输出] 描述符指针
    infiniopTensorDescriptor_t output_desc,              // [输入] 输出张量描述符
    infiniopTensorDescriptor_t input_desc                // [输入] 输入张量描述符
);
// 验证数据类型 (BF16/F16/F32/F64)，检查输入输出形状一致性，创建 ElementwiseInfo

// 获取工作空间大小
infiniStatus_t infiniopGetGeluWorkspaceSize(
    infiniopGeluDescriptor_t desc,                       // [输入] 描述符
    size_t *size                                         // [输出] 工作空间大小 (始终为 0)
);

// 执行 GELU 计算
infiniStatus_t infiniopGelu(
    infiniopGeluDescriptor_t desc,                       // [输入] 描述符
    void *workspace,                                     // [输入] 工作空间 (未使用)
    size_t workspace_size,                               // [输入] 工作空间大小
    void *output,                                        // [输出] 输出数据指针
    const void *input,                                   // [输入] 输入数据指针
    void *stream                                         // [输入] CPU 流 (未使用)
);
```

### 内部 C++ API
```cpp
namespace op::gelu::cpu {

// Descriptor 创建方法
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                            // CPU 设备句柄
    Descriptor **desc_ptr,                               // 输出描述符指针
    infiniopTensorDescriptor_t out_desc,                 // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec // 输入张量描述符向量
);
// 实现:
// 1. 检查数据类型是否为 BF16/F16/F32/F64
// 2. 验证输入输出形状完全相同 (CHECK_SAME_SHAPE)
// 3. 创建 ElementwiseInfo 元数据
// 4. 实例化 Descriptor 对象

// Descriptor 计算方法
infiniStatus_t Descriptor::calculate(
    void *workspace,                                     // 未使用
    size_t workspace_size,                               // 未使用
    void *output,                                        // 输出缓冲区
    std::vector<const void *> inputs,                    // 输入缓冲区向量 (单元素)
    void *stream                                         // 未使用
) const;
// 实现:
// 根据 _dtype 分发到对应的模板特化:
// - INFINI_DTYPE_BF16: 调用 _device_info->calculate<GeluOp, bf16_t>(...)
// - INFINI_DTYPE_F16:  调用 _device_info->calculate<GeluOp, fp16_t>(...)
// - INFINI_DTYPE_F32:  调用 _device_info->calculate<GeluOp, float>(...)
// - INFINI_DTYPE_F64:  调用 _device_info->calculate<GeluOp, double>(...)

}
```

## 4. 核心执行流程

### 4.1 初始化流程
```
1. 用户调用 infiniopCreateGeluDescriptor()
   ↓
2. operator.cc 根据 device 类型路由到 op::gelu::cpu::Descriptor::create()
   ↓
3. 创建流程:
   a. 类型检查: CHECK_DTYPE(dtype, BF16, F16, F32, F64)
   b. 形状验证: CHECK_SAME_SHAPE(output_shape, input_shape)
   c. 元数据生成: ElementwiseInfo::create(output_desc, {input_desc})
   d. 描述符实例化: new Descriptor(dtype, info, device_info, 0, device, device_id)
   ↓
4. 返回 Descriptor 指针给用户
```

### 4.2 计算执行流程
```
1. 用户调用 infiniopGelu(desc, workspace, output, input, stream)
   ↓
2. operator.cc 路由到 desc->calculate(workspace, workspace_size, output, {input}, stream)
   ↓
3. gelu_cpu.cc 中的 calculate() 根据 _dtype 分发:
   ↓
4. 调用 elementwise::cpu::DeviceImpl::calculate<GeluOp, T>(info, output, inputs, stream)
   ↓
5. 进入 calculate_impl 模板函数 (elementwise_cpu.h:153-181):
   ↓
6. 并行循环 (OpenMP):
   #pragma omp parallel for if (output_size > 1024)
   for (ptrdiff_t i = 0; i < output_size; ++i) {
       // 计算输出索引
       size_t out_idx = isOutputContiguous() ? i :
                        indexToOffset(i, ndim, output_shape, output_strides);

       // 计算输入索引
       size_t in_idx = isInputContiguous[0] ? i :
                       indexToOffset(i, ndim, input_shape, input_strides);

       // 执行 GELU 操作
       if constexpr (is_fp16_or_bf16) {
           out[out_idx] = cast<T>(GeluOp()(cast<float>(in[in_idx])));
       } else {
           out[out_idx] = GeluOp()(in[in_idx]);
       }
   }
   ↓
7. 返回 INFINI_STATUS_SUCCESS
```

### 4.3 索引计算细节
**连续内存优化**:
- 如果张量是连续的 (`isContiguous() == true`)，直接使用扁平索引 `i`
- 如果张量不连续或被广播，调用 `indexToOffset()` 将扁平索引转换为多维索引后再计算偏移量

**indexToOffset 算法** (`common_cpu.h`):
```cpp
// 将扁平索引 i 转换为多维索引 [d0, d1, ..., dn-1]
// 然后计算内存偏移: sum(di * strides[i])
size_t offset = 0;
for (int dim = ndim - 1; dim >= 0; --dim) {
    size_t idx = i % shape[dim];
    offset += idx * strides[dim];
    i /= shape[dim];
}
return offset;
```

## 5. 使用示例

```cpp
#include "infiniop/ops/gelu.h"

// 1. 创建张量描述符
size_t shape[] = {1024, 1024};
ptrdiff_t strides[] = {1024, 1};  // C 风格连续内存
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, 2, shape, strides, &input_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, 2, shape, strides, &output_desc);

// 2. 创建 GELU 描述符
infiniopGeluDescriptor_t gelu_desc;
infiniStatus_t status = infiniopCreateGeluDescriptor(
    handle,
    &gelu_desc,
    output_desc,
    input_desc
);

// 3. 分配内存
float *input = new float[1024 * 1024];
float *output = new float[1024 * 1024];
// 初始化 input 数据...

// 4. 获取工作空间 (GELU CPU 不需要工作空间)
size_t workspace_size = 0;
infiniopGetGeluWorkspaceSize(gelu_desc, &workspace_size);
void *workspace = nullptr;  // workspace_size == 0

// 5. 执行 GELU 计算
status = infiniopGelu(
    gelu_desc,
    workspace,
    workspace_size,
    output,
    input,
    nullptr  // CPU stream 未使用
);

// 6. 清理资源
infiniopDestroyGeluDescriptor(gelu_desc);
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
delete[] input;
delete[] output;
```

## 6. 实现细节

### 6.1 宏驱动架构
**ELEMENTWISE_DESCRIPTOR 宏** (`elementwise.h:15-54`):
- **设计模式**: 使用宏元编程自动生成统一的 Descriptor 类结构
- **优势**: 避免代码重复，确保所有逐元素操作 (ReLU, Sigmoid, GELU 等) 具有一致的接口
- **生成内容**:
  - 构造函数: 初始化基类 `InfiniopDescriptor` 和所有成员变量
  - 析构函数: 声明为纯虚函数，需在 .cc 文件中实现
  - `workspaceSize()`: 返回 `_workspace_size`
  - `create()`: 静态工厂方法声明
  - `calculate()`: 计算方法声明

### 6.2 类型派发机制
**类型分发表** (`gelu_cpu.cc:37-48`):
- 使用 `switch (_dtype)` 在运行时分发到正确的模板实例化
- 支持的派发类型:
  - `INFINI_DTYPE_BF16` → `bf16_t`
  - `INFINI_DTYPE_F16` → `fp16_t`
  - `INFINI_DTYPE_F32` → `float`
  - `INFINI_DTYPE_F64` → `double`
- 低精度类型 (FP16/BF16) 处理:
  - 先转换为 `float` 进行计算 (`utils::cast<float>`)
  - 调用 `GeluOp()` 计算 GELU 函数
  - 结果转换回原始类型 (`utils::cast<T>`)

### 6.3 并行化策略
**OpenMP 并行循环** (`elementwise_cpu.h:163`):
```cpp
#pragma omp parallel for if (output_size > 1024)
```
- **条件并行**: 仅当元素数 > 1024 时启用并行
- **调度策略**: 使用 OpenMP 默认的 static 调度
- **线程安全**: 每个线程处理独立的索引范围，无数据竞争
- **性能权衡**:
  - 小张量 (≤1024 元素): 串行执行，避免线程创建开销
  - 大张量 (>1024 元素): 并行执行，充分利用多核 CPU

### 6.4 内存管理
**元数据压缩存储** (`ElementwiseInfo`):
- 使用单一 `std::vector<size_t>` 存储所有元数据，减少内存分配次数
- 通过指针偏移和 `reinterpret_cast` 访问不同类型的数据区域
- **内存布局计算** (`elementwise.h:167-170`):
  ```cpp
  size_t meta_mem_size =
      ndim * (sizeof(size_t) + sizeof(ptrdiff_t)) +        // 输出形状和步长
      input_size * ndim * sizeof(size_t) +                  // 输入形状
      input_size * ndim * sizeof(ptrdiff_t) +               // 输入步长
      2 * input_size * sizeof(bool);                        // 连续性和广播标志
  ```

**智能指针管理**:
- `_device_info` 使用 `std::unique_ptr` 自动管理生命周期
- Descriptor 对象由用户手动管理 (`new`/`delete`)

### 6.5 误差函数计算
**std::erf 实现**:
- 使用 C++ 标准库的 `std::erf` 函数
- **数学定义**: `erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt`
- **数值特性**:
  - 输入范围: (-∞, +∞)
  - 输出范围: (-1, +1)
  - 渐近行为: `erf(x) → ±1` 当 `x → ±∞`
- **精度**: 依赖于 C++ 标准库实现，通常提供双精度 (float64) 级别的精度

**GELU 函数特性**:
- **单调性**: 严格单调递增
- **光滑性**: 无限可微
- **行为**:
  - `x → -∞`: GELU(x) → 0
  - `x → 0`: GELU(0) = 0
  - `x → +∞`: GELU(x) → x
- **神经网络中的优势**: 比 ReLU 更平滑，有助于梯度优化

### 6.6 错误处理
**验证机制**:
1. **数据类型检查** (`CHECK_DTYPE`):
   - 仅允许 BF16, F16, F32, F64
   - 其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

2. **形状一致性检查** (`CHECK_SAME_SHAPE`):
   - 输入和输出形状必须完全相同
   - 不允许广播 (broadcasting) 机制
   - 不匹配时返回错误状态码

3. **空指针检查**:
   - `ElementwiseInfo::create()` 检查描述符非空
   - 返回 `INFINI_STATUS_BAD_PARAM` 如果输入无效

### 6.7 性能优化技术
1. **连续内存路径**:
   - 检测 `isContiguous()` 标志
   - 连续张量使用扁平索引，避免 `indexToOffset` 开销

2. **类型特化**:
   - FP16/BF16 转换为 float 后统一计算
   - F32/F64 直接计算，避免类型转换

3. **编译期优化**:
   - 使用 `if constexpr` 在编译期分支，消除运行期条件判断
   - 模板内联: `GeluOp::operator()` 会被完全内联到循环中

4. **缓存友好性**:
   - 顺序访问模式 (sequential access)
   - 每个线程处理连续的内存块，提高缓存命中率

### 6.8 跨设备兼容性
**CPU 特定实现**:
- 不使用 CUDA/OpenCL 等加速器 API
- 仅依赖 OpenMP (可选) 进行多线程并行
- 可在任何支持 C++17 的 x86/ARM CPU 上编译运行

**与 CUDA 实现的对比**:
| 特性 | CPU 实现 | CUDA 实现 |
|------|---------|-----------|
| 并行原语 | OpenMP `#pragma omp parallel for` | CUDA kernel `<<<grid, block>>>` |
| 索引计算 | 运行期函数调用 `indexToOffset()` | 编译期常量或设备函数 |
| FP16/BF16 | `utils::cast` 转换为 float | `__half2float`/`__bfloat162float` 内置函数 |
| 内存模型 | 统一内存访问 | 分层内存 (global/shared/register) |
| 调度开销 | 线程池创建 (一次性) | Kernel 启动开销 (~10μs) |

## 7. 依赖关系

### 内部依赖
1. **逐元素操作框架** (`elementwise/cpu/elementwise_cpu.h`):
   - `ELEMENTWISE_DESCRIPTOR` 宏
   - `DeviceImpl` 类及其模板方法
   - `ElementwiseInfo` 元数据结构

2. **CPU 公共工具** (`devices/cpu/common_cpu.h`):
   - `indexToOffset()`: 多维索引计算

3. **通用工具** (`utils.h`, `utils/custom_types.h`):
   - `utils::cast<T>()`: 类型转换函数
   - `fp16_t`, `bf16_t`: 自定义 FP16/BF16 类型
   - `CEIL_DIV`: 整数除法向上取整宏

4. **张量抽象** (`tensor.h`):
   - `InfiniopTensorDescriptor`: 张量形状、步长、连续性查询

### 外部依赖
1. **C++ 标准库**:
   - `<cmath>`: `std::erf`, `std::sqrt`
   - `<vector>`: 动态数组
   - `<memory>`: `std::unique_ptr`
   - `<type_traits>`: `std::is_same_v`, `std::enable_if_t`

2. **OpenMP** (可选):
   - 仅在定义 `ENABLE_OMP` 时启用
   - 提供多线程并行支持

### 被依赖项
1. **GELU 操作统一接口** (`ops/gelu/operator.cc`):
   - C API `infiniopCreateGeluDescriptor`, `infiniopGelu` 等函数

2. **上层框架**:
   - InfiniCore 张量运算层
   - 深度学习算子库

## 8. 设计模式分析

### 8.1 策略模式 (Strategy Pattern)
- **上下文**: `Descriptor` 类
- **策略**: `GeluOp` 算子 (functor)
- **应用**: 将计算逻辑封装为可调用对象，支持不同激活函数的灵活替换

### 8.2 工厂模式 (Factory Pattern)
- **静态工厂方法**: `Descriptor::create()`
- **职责**: 验证参数、创建元数据、实例化对象
- **优势**: 封装创建复杂性，提供错误处理

### 8.3 模板方法模式 (Template Method Pattern)
- **骨架**: `DeviceImpl::calculate()` 定义执行流程
- **细节**: `GeluOp::operator()` 提供具体计算逻辑
- **扩展**: 通过模板参数 `Op` 支持不同算子

### 8.4 CRTP (奇异递归模板模式)
- **宏生成**: `ELEMENTWISE_DESCRIPTOR` 生成具有统一结构的类
- **多态支持**: 通过继承 `InfiniopDescriptor` 基类实现运行期多态
- **编译期优化**: 模板内联消除虚函数开销

## 9. 扩展指南

### 添加新的数据类型支持
在 `gelu_cpu.cc` 的 `calculate()` 方法中添加新的 case 分支:
```cpp
case INFINI_DTYPE_NEW_TYPE:
    return _device_info->calculate<GeluOp, new_type_t>(_info, output, inputs, stream);
```

### 实现其他激活函数
1. 创建新目录: `ops/new_op/cpu/`
2. 编写算子 functor:
   ```cpp
   struct NewOp {
       static constexpr size_t num_inputs = 1;  // 或其他输入数
       template <typename T>
       T operator()(const T &x) const {
           return /* 计算逻辑 */;
       }
   };
   ```
3. 使用宏生成 Descriptor: `ELEMENTWISE_DESCRIPTOR(new_op, cpu)`
4. 实现 `create()` 和 `calculate()` 方法 (类似 gelu_cpu.cc)

### 性能优化建议
1. **向量化**: 使用 SIMD 指令 (AVX2/AVX-512/NEON) 并行计算多个元素
2. **缓存分块**: 将大张量分块处理，提高缓存命中率
3. **NUMA 优化**: 在多路 CPU 服务器上绑定线程到 NUMA 节点
4. **数学近似**: 对 `erf()` 函数使用多项式近似，权衡精度和速度

## 10. 已知限制

1. **不支持广播**: 输入和输出形状必须完全一致
2. **不支持原地操作**: 必须提供独立的输入和输出缓冲区
3. **工作空间未使用**: `workspace` 参数保留供未来扩展
4. **Stream 未使用**: CPU 实现不支持异步执行流
5. **精度权衡**: FP16/BF16 类型通过 float 转换，可能损失部分精度

---

**文档版本**: 1.0
**最后更新**: 2025-01-14
**维护者**: InfiniCore Team
