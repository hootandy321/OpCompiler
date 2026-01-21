# `Embedding Operations` Core Implementation Documentation

该模块实现了词嵌入查找表操作(Embedding Lookup)，这是自然语言处理中将离散token索引映射到连续向量空间的基础算子。该实现支持CPU和设备(GPU等)两种后端，提供高效的内存拷贝机制实现索引查找。

## 1. Module Structure

- **`embedding.cc`**: 实现词嵌入查找操作的核心逻辑，包含内存分配、索引验证和向量拷贝算法
- **`embedding.hpp`** (位于 `include/infinicore/ops/`): 声明公共API接口，定义两个重载函数

## 2. Core Classes

### `embedding` Function
- **Location**: `embedding.cc:7-22`
- **Primary Function**: 实现带输出内存分配的词嵌入查找，根据输入索引张量从权重矩阵中提取对应的嵌入向量
- **Input Parameters**:
  - `input`: LongTensor类型，任意形状的张量，包含要提取的索引值
  - `weight`: 嵌入权重矩阵，浮点类型，形状为 `(V, embedding_dim)`，其中V为词汇表大小(最大索引+1)
- **Return Value**: 返回新分配的输出张量，形状为 `input_shape + [embedding_dim]`
- **Core Methods**:
  - `embedding(Tensor input, Tensor weight)`: 主函数，计算输出形状并分配内存，调用原地实现函数
- **Lifecycle**: 无状态函数，每次调用独立执行，无全局副作用

### `embedding_` Function (In-place)
- **Location**: `embedding.cc:24-87`
- **Primary Function**: 原地词嵌入查找实现，直接写入预分配的输出张量，支持多种数据类型和设备类型
- **Input Parameters**:
  - `out`: 预分配的输出张量，形状为 `input_shape + [embedding_dim]`
  - `input`: 索引张量，支持 `int64_t` 或 `int32_t` 类型
  - `weight`: 嵌入权重矩阵
- **Core Methods**:
  - `embedding_(Tensor out, Tensor input, Tensor weight)`: 原地执行查找，使用 `memcpy` 进行向量拷贝
- **Algorithm**: O(N) 时间复杂度，其中N为输入索引总数，对每个索引执行一次内存拷贝操作
- **Error Handling**: 使用断言验证索引范围 `(idx >= 0) && (idx < weight_shape[0])`，防止越界访问

## 3. API Interface

```cpp
namespace infinicore::op {

// 分配输出内存并执行词嵌入查找
Tensor embedding(Tensor input, Tensor weight);
// 功能：从权重矩阵中查找索引对应的嵌入向量
// 参数：
//   - input: 索引张量 (int32_t或int64_t类型)，任意形状
//   - weight: 嵌入矩阵，形状(V, embedding_dim)，V为词汇表大小
// 返回：新分配的输出张量，形状为 input_shape + [embedding_dim]

// 原地执行词嵌入查找（输出张量需预分配）
void embedding_(Tensor out, Tensor input, Tensor weight);
// 功能：直接将查找结果写入预分配的输出张量
// 参数：
//   - out: 预分配的输出张量，形状必须为 input_shape + [embedding_dim]
//   - input: 索引张量 (int32_t或int64_t类型)
//   - weight: 嵌入矩阵
// 返回：无返回值，结果直接写入out张量

} // namespace infinicore::op
```

## 4. Usage Example

```cpp
#include "infinicore/ops/embedding.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 示例：处理一批token索引，转换为嵌入向量
// 假设词汇表大小为10000，嵌入维度为768
const int vocab_size = 10000;
const int embedding_dim = 768;
const int batch_size = 32;
const int seq_length = 128;

// 创建权重矩阵：(10000, 768)
std::vector<Size> weight_shape = {vocab_size, embedding_dim};
Tensor weight = Tensor::randn(weight_shape, DataType::F32, Device::cpu());

// 创建输入索引张量：(32, 128)，包含token ID
std::vector<Size> input_shape = {batch_size, seq_length};
Tensor input = Tensor::randint(input_shape, 0, vocab_size, DataType::I64, Device::cpu());

// 方法1：使用自动分配内存的版本
Tensor inputs_embeds = op::embedding(input, weight);
// inputs_embeds 形状：(32, 128, 768)

// 方法2：使用原地版本（需预分配输出张量）
Tensor output = Tensor::empty({batch_size, seq_length, embedding_dim},
                              DataType::F32, Device::cpu());
op::embedding_(output, input, weight);
// output 现在包含嵌入向量

// 注意事项：
// 1. 索引值必须在 [0, vocab_size) 范围内，否则触发断言失败
// 2. 输入张量仅支持 CPU 设备
// 3. 权重张量可以是 CPU 或其他设备（GPU等）
```

## 5. Implementation Details

- **内存管理**:
  - **输出分配**: `embedding()` 函数使用 `Tensor::empty()` 在目标设备上分配连续内存
  - **内存布局**: 输出张量按行主序存储，每个嵌入向量连续存储
  - **内存拷贝**: 使用标准库 `std::memcpy()` (CPU) 或 `context::memcpyD2D()` (设备间) 进行高效拷贝

- **算法细节**:
  - **索引查找**: 对每个输入索引，计算权重矩阵中的偏移量 `idx * bytes`，直接拷贝整个嵌入向量
  - **批处理**: 展平输入张量的所有维度，统一处理，支持任意形状的输入
  - **复杂度**: 时间复杂度 O(N * D)，其中N为token数量，D为嵌入维度；空间复杂度 O(N * D) 用于输出

- **数据类型支持**:
  - **输入索引**: 支持 `int32_t` (DataType::I32) 和 `int64_t` (DataType::I64)
  - **权重和输出**: 支持所有浮点类型 (F32, F16, BF16等)，由 `dsize()` 函数动态计算字节数

- **设备支持**:
  - **输入约束**: 输入张量必须在 CPU 设备上（断言检查）
  - **权重灵活性**: 权重张量可在 CPU 或其他设备（如 GPU）
  - **设备间拷贝**: 当权重在非CPU设备时，使用 `context::memcpyD2D()` 进行设备间内存传输

- **错误处理**:
  - **索引越界**: 使用 `assert()` 验证每个索引在有效范围内 `[0, vocab_size)`
  - **类型检查**: 断言输入张量类型必须是 I32 或 I64
  - **设备检查**: 断言输入张量必须在 CPU 设备上

- **性能优化**:
  - **批量拷贝**: 每次拷贝整个嵌入向量的连续内存块，减少循环次数
  - **条件分支**: 编译时确定数据类型和设备类型，避免运行时类型判断开销
  - **内存对齐**: 使用 `std::memcpy` 保证对齐拷贝，提升 CPU 缓存命中率

- **依赖关系**:
  - **内部依赖**: `context.hpp` (设备内存拷贝), `tensor.hpp` (张量数据结构)
  - **标准库**: `<cstring>` (提供 std::memcpy)
  - **类型系统**: 依赖 `DataType`, `Device`, `Size`, `Tensor` 等核心类型定义

- **设计模式**:
  - **双重API**: 提供分配版本(`embedding`)和原地版本(`embedding_`)，灵活控制内存管理
  - **模板式条件编译**: 通过 if-else 分支处理不同数据类型和设备类型，避免模板膨胀
  - **防御性编程**: 使用断言在开发阶段捕获非法输入，生产环境可编译去除

- **线程安全**: 该实现是无状态的纯函数，不修改共享全局状态，理论上支持并行调用，但底层 `memcpy` 和设备拷贝函数的线程安全性取决于上下文实现
