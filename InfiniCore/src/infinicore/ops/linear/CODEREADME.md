# `Linear Operation` 核心实现文档

该模块实现了神经网络中的全连接层（线性变换）操作，支持任意维度的输入张量，通过矩阵乘法（GEMM）和可选的偏置项完成特征变换。核心采用视图重塑（view reshaping）技术将高维输入展平为二维矩阵，然后调用底层优化的 GEMM 算子完成计算。

## 1. 模块结构

- **`linear.cc`**: 线性变换操作的核心实现，包含内存分配和原位计算两个版本的函数

## 2. 核心函数

### `linear(Tensor input, Tensor weight, std::optional<Tensor> bias)`
- **位置**: `linear.cc:7-22`
- **主要功能**: 分配输出内存并执行线性变换的非原位版本
- **参数**:
  - `input`: 输入张量，支持任意维度（N维）
  - `weight`: 权重矩阵，形状为 `[out_features, in_features]`
  - `bias`: 可选的偏置向量，形状为 `[out_features]`，使用 `std::optional` 包装
- **返回值**: 返回输出张量，形状除最后一维外与输入相同，最后一维为 `out_features`
- **核心算法**:
  1. 提取输入张量维度数 `ndim` 和权重输出特征数 `out_features`
  2. 构造输出形状：复制输入形状，将最后一维替换为 `out_features`
  3. 使用 `Tensor::empty()` 在输入张量的设备和数据类型上分配输出内存
  4. 调用原位版本 `linear_()` 完成计算
  5. 返回输出张量
- **时间复杂度**: O(N × in_features × out_features)，其中 N 为展平后的批量大小
- **空间复杂度**: O(N × out_features) 用于输出存储

### `linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias)`
- **位置**: `linear.cc:24-57`
- **主要功能**: 在预分配的输出张量上执行原位线性变换计算
- **参数**:
  - `out`: 预分配的输出张量，调用者负责确保内存正确分配
  - `input`: 输入张量
  - `weight`: 权重矩阵
  - `bias`: 可选的偏置项
- **核心算法流程**:
  1. **形状提取与验证**:
     - 从权重矩阵提取 `out_features` 和 `in_features`
     - 断言输出张量维度数与输入一致

  2. **批量维度计算** (第 36-41 行):
     - 计算除最后一维外所有维度的乘积作为批量大小 `N`
     - 例如：输入 `[batch, seq_len, in_features]` → `N = batch × seq_len`

  3. **视图重塑** (第 44 行):
     - 将输出张量重塑为二维视图 `[N, out_features]`
     - 利用 `Tensor::view()` 实现零拷贝的形状变换

  4. **偏置广播** (第 46-52 行):
     - 如果存在偏置项，使用 `rearrange_()` 将偏置广播到 `[N, out_features]`
     - 通过 `as_strided({N, out_features}, {0, 1})` 实现零拷贝广播，步长为 `{0, 1}` 表示沿第一维重复，第二维连续
     - 设置 GEMM 的 `beta` 参数为 1.0，使 GEMM 结果累加到偏置上

  5. **矩阵乘法** (第 54-56 行):
     - 调用 `gemm_()` 执行核心计算：`output = alpha × input @ weight^T + beta × output`
     - 输入重塑为 `[N, in_features]`
     - 权重转置为 `[in_features, out_features]`（原始存储为 `[out_features, in_features]`）
     - `alpha = 1.0` 为缩放因子
     - `beta` 根据偏置存在性设为 0.0（无偏置）或 1.0（有偏置）
- **内存效率**: 使用视图机制避免数据复制，所有形状变换均为零拷贝操作
- **依赖关系**:
  - `gemm_()`: 底层优化的通用矩阵乘法内核
  - `rearrange_()`: 张量重排操作，用于广播
  - `Tensor::view()`, `Tensor::permute()`, `Tensor::as_strided()`: 张量视图操作

## 3. API 接口

```cpp
namespace infinicore::op {

// 非原位线性变换：自动分配输出内存
Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias);
// 返回新的输出张量，形状为 [..., out_features]

// 原位线性变换：在预分配张量上计算
void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias);
// 直接写入 out，无返回值

} // namespace infinicore::op
```

## 4. 使用示例

```cpp
#include "infinicore/ops/linear.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;
using namespace infinicore::op;

// 示例 1: 简单的二维输入（全连接层）
// 输入: [batch_size=32, in_features=128]
// 输出: [batch_size=32, out_features=256]
auto input = Tensor::empty({32, 128}, DataType::FLOAT32, Device::cpu());
auto weight = Tensor::empty({256, 128}, DataType::FLOAT32, Device::cpu());
auto bias = Tensor::empty({256}, DataType::FLOAT32, Device::cpu());

// 非原位调用：自动分配输出
auto output = linear(input, weight, bias);
// output 形状: [32, 256]

// 示例 2: 高维输入（如 Transformer 的 token 嵌入）
// 输入: [batch=4, seq_len=128, in_features=768]
auto input_3d = Tensor::empty({4, 128, 768}, DataType::FLOAT32, Device::cpu());
auto weight_3d = Tensor::empty({1024, 768}, DataType::FLOAT32, Device::cpu());

// 无偏置的线性变换
auto output_3d = linear(input_3d, weight_3d, std::nullopt);
// 内部计算: reshape 为 [512, 768] × [768, 1024] = [512, 1024]
// 输出形状: [4, 128, 1024]

// 示例 3: 原位调用（手动管理内存）
auto input_manual = Tensor::empty({16, 64}, DataType::FLOAT32, Device::cpu());
auto weight_manual = Tensor::empty({128, 64}, DataType::FLOAT32, Device::cpu());
auto bias_manual = Tensor::empty({128}, DataType::FLOAT32, Device::cpu());

// 预分配输出内存
Shape out_shape = {16, 128};
auto output_manual = Tensor::empty(out_shape, DataType::FLOAT32, Device::cpu());

// 原位计算
linear_(output_manual, input_manual, weight_manual, bias_manual);
// output_manual 现在包含计算结果
```

## 5. 实现细节

- **内存管理策略**:
  - 非原位版本使用 `Tensor::empty()` 自动分配输出张量，遵循 RAII 语义
  - 原位版本由调用者负责输出张量的生命周期管理
  - 所有中间视图操作（`view()`, `permute()`, `as_strided()`）均为零拷贝，不产生新的内存分配

- **性能优化技术**:
  1. **视图融合**: 通过 `view()` 将高维输入展平为二维，避免显式的 `reshape` 复制操作
  2. **权重转置**: 在 GEMM 调用时使用 `permute({1, 0})` 实现逻辑转置，某些后端可能优化为就地计算
  3. **广播优化**: 使用 `as_strided({N, out_features}, {0, 1})` 实现零拷贝的偏置广播，步长 0 表示维度重复
  4. **GEMM 参数控制**: 通过 `alpha` 和 `beta` 参数实现条件累加，避免显式的加法操作
     - 无偏置: `beta = 0.0`, GEMM 直接写入输出
     - 有偏置: `beta = 1.0`, GEMM 结果累加到已广播的偏置上

- **并发性**:
  - 线程安全由底层的 `gemm_()` 和 `rearrange_()` 操作保证
  - 该模块本身不引入额外的同步原语

- **错误处理**:
  - 使用 `assert()` 验证输出张量维度数（第 34 行），仅在调试模式下生效
  - 依赖底层操作（GEMM、rearrange）进行形状和类型合法性检查

- **依赖项**:
  - **外部依赖**:
    - `infinicore/ops/gemm.hpp`: 通用矩阵乘法后端
    - `infinicore/ops/rearrange.hpp`: 张量重排和广播操作
    - `infinicore/tensor.hpp`: 核心张量抽象，提供 `empty()`, `view()`, `permute()`, `as_strided()` 等方法
  - **标准库**: `<optional>` 用于可选偏置参数

- **设计模式**:
  1. **双重接口模式**: 提供 `linear()`（分配+计算）和 `linear_()`（原位计算）两种版本，满足灵活性和性能需求
  2. **零拷贝视图**: 利用张量视图系统避免中间结果复制
  3. **惰性求值**: 偏置广播通过 `as_strided()` 延迟到 GEMM 内核处理

- **算法复杂度**:
  - **时间复杂度**: O(N × in_features × out_features)，由 GEMM 主导
  - **空间复杂度**:
    - 非原位版本: O(N × out_features) 输出存储
    - 原位版本: O(1) 额外空间（视图操作不分配内存）
  - **批量大小计算**: O(ndim) 遍历输入形状计算 N，通常 ndim ≤ 4，开销可忽略

- **数值稳定性**:
  - 直接委托给底层 GEMM 实现，稳定性取决于后端数值算法
  - `alpha = 1.0`, `beta = 1.0` 确保无额外缩放误差

- **设备兼容性**:
  - 通过 `Tensor::empty()` 和底层操作自动适配设备（CPU/CUDA/其他加速器）
  - 权重转置和视图操作在不同设备上的语义保持一致
