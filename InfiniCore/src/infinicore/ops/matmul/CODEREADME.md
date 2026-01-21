# `matmul` 矩阵乘法运算模块核心实现文档

本模块实现矩阵乘法运算（Matrix Multiplication），作为通用线性代数运算的基础组件。它是对 GEMM（General Matrix Multiply）运算的简化封装，提供直观的矩阵乘法接口，广泛应用于神经网络层的矩阵变换、线性映射等场景。

## 1. 模块结构

- **`matmul.cc`**: 矩阵乘法运算的轻量级实现，通过委托给 GEMM 算子实现核心功能
- **`matmul.hpp`**: 公共 API 接口定义，声明了矩阵乘法的前向和原地操作函数

## 2. 核心类与函数

本模块采用函数式设计，不定义独立类，而是提供两个核心 API 函数：

### `matmul` - 矩阵乘法（返回新张量）

- **位置**: `matmul.cc:6-8`
- **函数签名**: `Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f)`
- **主要功能**: 执行矩阵乘法运算 `C = alpha * (A × B)`，返回新分配的结果张量
- **参数说明**:
  - `a`: 左矩阵张量（输入），支持任意维度（最后两维参与矩阵乘法）
  - `b`: 右矩阵张量（输入），支持任意维度（最后两维参与矩阵乘法）
  - `alpha`: 标量缩放因子，默认值为 1.0f，用于对乘积结果进行缩放
- **返回值**: 新分配的张量，形状为 `a.shape[:-2] + (a.shape[-2], b.shape[-1])`
- **实现细节**:
  - 直接委托给 `gemm` 函数，传入 `beta = 0.0f`（表示不累加到现有输出）
  - 利用 GEMM 的输出自动分配机制，避免手动管理内存
  - 计算复杂度为 O(m × n × k)，其中 m, k 为 A 的行/列数，n 为 B 的列数

### `matmul_` - 原地矩阵乘法（In-place）

- **位置**: `matmul.cc:10-12`
- **函数签名**: `void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f)`
- **主要功能**: 执行矩阵乘法运算 `C = alpha * (A × B)`，结果写入预分配的张量 C
- **参数说明**:
  - `c`: 输出张量（预先分配），必须与 a、b 形状兼容且位于同一设备
  - `a`: 左矩阵张量（输入）
  - `b`: 右矩阵张量（输入）
  - `alpha`: 标量缩放因子，默认值为 1.0f
- **返回值**: 无（直接写入张量 c）
- **实现细节**:
  - 直接调用 `Gemm::execute` 静态方法触发图算子执行
  - 支持图模式录制（通过 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 宏）
  - 适用于需要复用输出内存的高性能场景

## 3. API 接口

```cpp
namespace infinicore::op {

// 矩阵乘法：返回新分配的结果张量
// 计算：C = alpha * (A @ B)
Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f);

// 原地矩阵乘法：结果写入预分配的张量 c
// 计算：C = alpha * (A @ B)
void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f);

} // namespace infinicore::op
```

**接口设计说明**：
1. **命名约定**: 遵循 InfiniCore 的下划线后缀约定，`matmul_` 表示原地操作（in-place）
2. **默认参数**: `alpha` 默认为 1.0f，符合大多数矩阵乘法场景
3. **设备约束**: 输入输出张量必须位于同一设备（由底层 GEMM 算子通过 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏强制检查）
4. **数据类型**: 支持所有浮点类型（F16, F32, F64, BF16）以及整数类型（具体支持取决于底层后端实现）

## 4. 使用示例

### 示例 1：基本矩阵乘法（返回新张量）

```cpp
#include "infinicore/ops/matmul.hpp"

using namespace infinicore;

// 创建两个 2D 张量
// A: shape (3, 4), B: shape (4, 5)
Tensor A = Tensor::empty({3, 4}, DataType::F32, Device::cpu());
Tensor B = Tensor::empty({4, 5}, DataType::F32, Device::cpu());

// 填充数据（示例）
std::fill_n((float*)A->data(), 12, 1.0f);  // A 全为 1
std::fill_n((float*)B->data(), 20, 2.0f);  // B 全为 2

// 执行矩阵乘法：C = A @ B
// 结果 C 的 shape 为 (3, 5)，每个元素为 8.0 (1*2*4)
Tensor C = op::matmul(A, B);

// 使用 alpha 缩放：C = 0.5 * (A @ B)
Tensor C_scaled = op::matmul(A, B, 0.5f);  // 结果为 4.0
```

### 示例 2：批量矩阵乘法（Batched MatMul）

```cpp
// 创建批次张量
// A: shape (2, 3, 4) - 2 个 3x4 矩阵
// B: shape (2, 4, 5) - 2 个 4x5 矩阵
Tensor A_batch = Tensor::empty({2, 3, 4}, DataType::F32, Device::cpu());
Tensor B_batch = Tensor::empty({2, 4, 5}, DataType::F32, Device::cpu());

// 执行批量矩阵乘法
// 结果 C_batch: shape (2, 3, 5)
// 每个批次独立执行：C_batch[i] = A_batch[i] @ B_batch[i]
Tensor C_batch = op::matmul(A_batch, B_batch);
```

### 示例 3：原地操作（复用内存）

```cpp
// 预分配输出张量（避免内存分配开销）
Tensor A = Tensor::zeros({3, 4}, DataType::F32, Device::cpu());
Tensor B = Tensor::ones({4, 5}, DataType::F32, Device::cpu());
Tensor C = Tensor::empty({3, 5}, DataType::F32, Device::cpu());

// 原地计算：结果直接写入 C
op::matmul_(C, A, B, 1.0f);

// 后续可以复用 C 的内存进行其他计算
op::matmul_(C, A, B, 2.0f);  // C = 2 * (A @ B)，覆盖旧值
```

### 示例 4：神经网络线性层模拟

```cpp
// 模拟线性层：y = xW^T + b（bias 需单独加）
Tensor x = Tensor::empty({128, 64}, DataType::F32, Device::cpu());    // 输入：batch=128, in_dim=64
Tensor W = Tensor::empty({32, 64}, DataType::F32, Device::cpu());     // 权重：out_dim=32, in_dim=64
Tensor bias = Tensor::empty({32}, DataType::F32, Device::cpu());       // 偏置：out_dim=32

// 计算矩阵乘法部分：y = x @ W^T
// 注意：matmul 会使用 W 的最后两维，所以需要确保形状匹配
Tensor y = op::matmul(x, W, 1.0f);  // y: shape (128, 32)

// 添加偏置（使用 add 算子）
op::add_(y, y, bias.unsqueeze(0));  // 广播加法
```

### 示例 5：多硬件后端支持（CUDA 示例）

```cpp
// 在 NVIDIA GPU 上执行矩阵乘法
Device gpu_device(Device::Type::NVIDIA, 0);  // 使用第一个 NVIDIA GPU

Tensor A_gpu = Tensor::ones({1024, 1024}, DataType::F16, gpu_device);
Tensor B_gpu = Tensor::ones({1024, 1024}, DataType::F16, gpu_device);

// 自动调度到 CUDA 后端执行
Tensor C_gpu = op::matmul(A_gpu, B_gpu);

// 如果需要将结果移回 CPU
Tensor C_cpu = C_gpu.to(Device::cpu());
```

## 5. 实现细节

### 5.1 设计模式与架构

- **委托模式 (Delegation Pattern)**: `matmul` 不直接实现矩阵乘法逻辑，而是委托给更底层的 `gemm` 算子。这种设计实现了代码复用和职责分离：
  ```cpp
  // matmul.cc:6-8
  Tensor matmul(Tensor a, Tensor b, float alpha) {
      return gemm(a, b, alpha, 0.0f);  // beta=0 表示不累加
  }
  ```

- **零拷贝抽象**: 通过函数式 API 隐藏底层 GEMM 算子的复杂性，用户无需了解图算子、调度器等实现细节

- **统一接口命名**: 遵循 InfiniCore 的命名规范，与其他运算符（如 `add`, `mul`）保持一致

### 5.2 与 GEMM 的关系

GEMM（General Matrix Multiply）是 BLAS 标准中的核心函数，定义为：
```
C = alpha * op(A) @ op(B) + beta * C
```

`matmul` 是 GEMM 的简化版本，固定参数：
- `alpha`: 用户可配置（默认 1.0）
- `beta`: 固定为 0.0（不累加到 C，而是覆盖）
- `op(A)` 和 `op(B)`: 固定为不转置（标准矩阵乘法）

**参数映射**：
```cpp
matmul(a, b, alpha)  等价于  gemm(a, b, alpha, 0.0)
```

### 5.3 设备调度机制

虽然 `matmul.cc` 本身不包含设备相关代码，但通过 GEMM 算子实现了跨设备调度：

1. **设备检查**: 在 `Gemm` 构造函数中，`INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b)` 确保所有张量位于同一设备
2. **动态分发**: `INFINICORE_GRAPH_OP_DISPATCH` 宏根据张量的设备类型（CPU/CUDA/Ascend 等）选择对应的内核实现
3. **后端支持**: 通过注册机制支持 9 种硬件后端（NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGON、QY）

### 5.4 内存管理策略

- **输出分配（`matmul` 函数）**:
  - 由 `gemm` 函数内部调用 `Tensor::empty` 自动分配
  - 形状推断：取 `a` 的形状，将最后一维替换为 `b` 的最后一维
  - 设备继承：输出张量与输入张量位于同一设备

- **原地操作（`matmul_` 函数）**:
  - 要求用户预先分配输出张量 `c`
  - 避免额外的内存分配开销，适合高频调用场景
  - 用户需确保 `c` 的形状和类型正确

### 5.5 图模式支持

通过 `Gemm::execute` 内部的 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 宏，支持两种执行模式：

1. **立即执行模式 (Eager Mode)**:
   - 默认模式，直接调用硬件内核执行计算
   - 适用于交互式开发和小规模运算

2. **图录制模式 (Graph Mode)**:
   - 当 `context::isGraphRecording()` 为真时，将算子添加到计算图
   - 延迟执行，支持算子融合、内核优化等
   - 适用于生产环境和高性能推理

### 5.6 形状广播规则

`matmul` 支持高维张量的批量矩阵乘法，遵循 NumPy/PyTorch 的广播语义：

```cpp
// 规则：最后两维参与矩阵乘法，前面的维度必须相同或可广播
A: (..., M, K)
B: (..., K, N)
结果: (..., M, N)

// 示例：
A: (2, 3, 4, 5)  -> 最后两维 (4, 5)
B: (2, 3, 5, 6)  -> 最后两维 (5, 6)
C: (2, 3, 4, 6)  -> 最后两维 (4, 6)

// 广播示例：
A: (1, 3, 4)     -> 可广播为 (2, 3, 4)
B: (2, 3, 4, 5)  -> 保持不变
C: (2, 3, 3, 5)  -> 批次维度广播
```

### 5.7 数据类型支持

支持 InfiniCore 定义的所有数值类型：

- **浮点类型**: F16, F32, F64, BF16（BFloat16）- 适用于深度学习
- **整数类型**: I8, I16, I32, I64, U8, U16, U32, U64 - 适用于传统数值计算
- **复杂类型**: C16, C32, C64, C128 - 适用于信号处理

**注意**：具体支持的数据类型取决于底层硬件后端的实现能力。

### 5.8 性能考虑

- **计算复杂度**: O(m × n × k)，其中 m, k 为 A 的维度，n 为 B 的维度
- **空间局部性**: GEMM 是计算密集型操作，通常受限于内存带宽
- **优化策略**: 底层后端可能使用以下优化：
  - 分块 (Tiling) 以提高缓存命中率
  - 向量化 (SIMD/AVX) 指令加速
  - 多线程并行 (OpenMP/pthreads)
  - Tensor Core（NVIDIA GPU）或矩阵加速单元（ASIC）

### 5.9 错误处理

通过底层 GEMM 算子提供的错误检查：

1. **设备不匹配**:
   ```cpp
   Tensor A = Tensor::empty({3, 4}, DataType::F32, Device::cpu());
   Tensor B = Tensor::empty({4, 5}, DataType::F32, Device(Device::Type::NVIDIA));
   Tensor C = op::matmul(A, B);  // 抛出 std::runtime_error
   ```

2. **形状不兼容**:
   ```cpp
   Tensor A = Tensor::empty({3, 4}, DataType::F32, Device::cpu());
   Tensor B = Tensor::empty({5, 6}, DataType::F32, Device::cpu());  // 维度 4 != 5
   Tensor C = op::matmul(A, B);  // 底层内核返回错误
   ```

3. **类型不支持**:
   ```cpp
   Tensor A = Tensor::empty({3, 4}, DataType::BOOL, Device::cpu());
   Tensor B = Tensor::empty({4, 5}, DataType::BOOL, Device::cpu());
   Tensor C = op::matmul(A, B);  // 某些后端可能不支持布尔矩阵乘法
   ```

### 5.10 依赖关系

**模块依赖**：
- `infinicore/ops/matmul.hpp` - 公共接口
- `infinicore/ops/gemm.hpp` - 底层 GEMM 算子
- `infinicore/tensor.hpp` - 张量类型定义
- `infinicore/device.hpp` - 设备类型枚举
- `infinicore/dtype.hpp` - 数据类型枚举
- `infinicore/graph/graph.hpp` - 图算子基础设施

**反向依赖**（上层模块）：
- 神经网络层（线性层、注意力机制等）
- 优化器（梯度计算）
- 数学库（线性方程求解、特征值分解等）

### 5.11 扩展性

本模块设计为极简封装，扩展功能建议通过以下方式：

1. **自定义 alpha 缩放**:
   ```cpp
   Tensor C = op::matmul(A, B, 0.5f);  // 半缩放
   ```

2. **累加模式**（需要直接使用 GEMM）:
   ```cpp
   // C = 2.0 * (A @ B) + 3.0 * C
   op::gemm_(C, A, B, 2.0f, 3.0f);
   ```

3. **转置支持**（需要直接使用 GEMM 或张量变换）:
   ```cpp
   // C = A @ B^T
   Tensor B_T = B.permute({1, 0});  // 手动转置
   Tensor C = op::matmul(A, B_T);
   ```

4. **融合操作**（建议使用图模式）:
   ```cpp
   // 在图模式下自动融合 matmul + bias + activation
   context::beginGraphRecord();
   Tensor y = op::matmul(x, W);
   op::add_(y, y, bias);
   op::gelu_(y, y);  // 假设有 gelu 算子
   auto graph = context::endGraphRecord();
   graph->run();  // 融合执行
   ```

## 总结

`matmul` 模块通过简洁的 API 封装了复杂的矩阵乘法实现，隐藏了设备调度、内存管理、图模式等底层细节。作为 InfiniCore 线性代数运算的基础组件，它为上层神经网络层提供了高效的矩阵变换能力，同时通过委托给 GEMM 算子实现了跨硬件后端的无缝支持。其设计体现了"简单接口，复杂实现"的工程哲学。
