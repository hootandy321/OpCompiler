# Common Utilities Core Implementation Documentation

本模块提供 InfiniTrain 框架示例程序所需的基础工具类，包括数据集加载器（Tiny Shakespeare）、分词器（Tokenizer）和二进制文件读取工具。这些组件共同支撑了语言模型训练和推理的数据预处理、模型权重加载和文本生成流程。

## 1. Module Structure

- **`tiny_shakespeare_dataset.h/cc`**: Tiny Shakespeare 文本数据集加载器，支持从自定义二进制格式读取分词后的文本数据，提供 GPT-2 和 LLaMA 3 两种 token 编码格式
- **`tokenizer.h/cc`**: 分词器实现，负责读取词汇表文件、将 token ID 解码为文本字符串，以及实现自回归文本生成（autoregressive generation）
- **`utils.h/cc`**: 二进制文件读取工具集，提供 BF16/FP32 转换、矩阵和向量的分片读取功能，支持高效的模型参数加载

## 2. Core Classes

### `TinyShakespeareDataset`
- **Location**: `tiny_shakespeare_dataset.h`, `tiny_shakespeare_dataset.cc`
- **Primary Function**: 实现 `infini_train::Dataset` 接口，从预处理后的二进制文件加载 Tiny Shakespeare 文本数据集，为语言模型训练提供序列数据。该数据集通过滑动窗口方式生成输入序列 x 和目标序列 y（y 为 x 的下一个 token）。
- **Key Members**:
  - `text_file_`: `TinyShakespeareFile` 结构体，存储从文件读取的元数据（类型、维度）和完整的 token 张量
  - `sequence_length_`: `const size_t`，每个序列的 token 数量（例如 256）
  - `sequence_size_in_bytes_`: `const size_t`，单个序列的字节大小，用于计算偏移量
  - `num_samples_`: `const size_t`，可生成的样本数量（总 token 数 / sequence_length - 1）
- **Core Methods**:
  - `TinyShakespeareDataset(filepath, sequence_length)`: 构造函数，调用 `ReadTinyShakespeareFile` 读取二进制文件，验证文件格式，计算样本数量
  - `operator[](idx)`: 返回第 `idx` 个训练样本，包含输入序列 `x` 和目标序列 `y`，两者均为 `(sequence_length,)` 形状的 Tensor。通过字节偏移实现零拷贝视图（zero-copy view）
  - `Size()`: 返回数据集样本总数
- **Lifecycle**: 在构造时一次性加载整个数据集到内存，通过 `operator[]` 提供只读视图访问。由于使用 `Tensor` 的共享指针构造函数（offset-based view），不会产生数据复制

### `TinyShakespeareFile` (Nested Struct)
- **Location**: `tiny_shakespeare_dataset.h`
- **Primary Function**: 封装二进制文件的元数据和数据内容
- **Key Members**:
  - `type`: `TinyShakespeareType` 枚举，指示文件使用 UINT16（GPT-2）或 UINT32（LLaMA 3）编码
  - `dims`: `std::vector<int64_t>`，数据形状，为 `[num_sequences, sequence_length]`
  - `tensor`: `infini_train::Tensor`，存储所有 token 的 INT64 张量

### `Tokenizer`
- **Location**: `tokenizer.h`, `tokenizer.cc`
- **Primary Function**: 管理词汇表（vocabulary table），提供 token ID 到字符串的解码功能，并实现基于概率分布的自回归文本生成。使用预定义的提示词（如 "The meaning of life is"）作为上下文，通过 multinomial sampling 逐个生成 token
- **Key Members**:
  - `magic_number_`: `uint32_t`，从文件头读取的魔数，用于识别分词器类型（GPT-2: 20240328, LLaMA-3: 20240801）
  - `vocab_size_`: `uint32_t`，词汇表大小
  - `token_table_`: `std::vector<std::string>`，存储每个 token ID 对应的文本字符串
  - `eot_token_`: `uint32_t`，End-of-Token 标记（GPT-2: 50256, LLaMA-3: 128001）
- **Core Methods**:
  - `Tokenizer(filepath)`: 构造函数，读取 1024 字节文件头（魔数、版本、vocab size），然后读取 `vocab_size` 个变长字符串（每个前缀为 1 字节长度）构建词汇表
  - `Decode(token_id)`: 将 token ID 映射为对应的文本字符串，越界返回 "[INVALID_TOKEN]"
  - `GenerateText(model, batch_size, sequence_length, text_length, device)`: 执行自回归文本生成，初始化输入张量为 EOT token，插入预定义提示词，循环调用 `model.Forward` 获取 logits，应用 Softmax 后使用 `SampleMult` 进行采样，将生成的 token 添加到输入序列并打印
- **Lifecycle**: 在构造时加载整个词汇表到内存，`GenerateText` 方法会动态创建 Tensor 并在 CPU 和计算设备之间传输数据

### Random Number Generation (Internal Functions)
- **Location**: `tokenizer.cc`
- **Primary Function**: 实现基于 Xorshift 算法的伪随机数生成器，用于文本生成时的 token 采样
- **Core Methods**:
  - `RandomU32(state)`: 使用 Xorshift* 算法生成 32 位随机整数，状态更新为 `state ^= state >> 12; state ^= state << 25; state ^= state >> 27; return (state * 0x2545F4914F6CDD1Dull) >> 32`
  - `RandomF32(state)`: 生成 [0, 1) 范围内的随机浮点数，通过 `(RandomU32(state) >> 8) / 2^24` 实现
  - `SampleMult(probabilities, n, coin)`: 根据概率分布进行多项式采样，计算累积分布函数（CDF），找到第一个使得 `coin < cdf` 的索引

## 3. API Interface

```cpp
// Dataset Interface for Loading Tiny Shakespeare Data
class TinyShakespeareDataset : public infini_train::Dataset {
public:
    enum class TinyShakespeareType : int { kUINT16, kUINT32, kINVALID };

    TinyShakespeareDataset(const std::string &filepath, size_t sequence_length);
    // 读取二进制文件，验证格式，加载所有 tokens 到内存

    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t idx) const override;
    // 返回 (x, y) 对，x 为输入序列，y 为目标序列（x 的下一个 token）

    size_t Size() const override;
    // 返回数据集样本数量
};

// Tokenizer Interface for Text Decoding and Generation
namespace infini_train {
class Tokenizer {
public:
    enum class Version : uint32_t { kV1 = 1, kV2 = 2 };

    Tokenizer(const std::string &filepath);
    // 从二进制文件加载词汇表

    std::string Decode(uint32_t token_id) const;
    // 将 token ID 转换为文本字符串

    void GenerateText(nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                      uint32_t text_length, const Device *device) const;
    // 使用指定模型执行自回归文本生成，从预定义提示词开始生成指定长度的文本

    uint32_t GetEndToken() const;
    // 返回 EOT token ID
};
}

// Binary File Reading Utilities
namespace infini_train {
float ConvertBF16ToFloat(void *ptr);
// 将 Brain Float 16（BF16）转换为 IEEE 754 Float32，通过左移 16 位实现

template <typename T>
T BytesToType(const std::vector<uint8_t> &bytes, size_t offset);
// 从字节数组的指定偏移量处反序列化一个平凡可复制类型（trivially copyable type）

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs);
// 从输入流读取指定字节数

void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols);
// 读取完整的 row-major 矩阵（rows × cols 个 float32）

void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols,
                             int64_t row_start, int64_t row_cnt);
// 读取矩阵的行分片：[row_start : row_start+row_cnt) × [0:cols)

void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols,
                             int64_t col_start, int64_t col_cnt);
// 读取矩阵的列分片：[0:rows) × [col_start : col_start+col_cnt)，逐行跳跃读取

void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len);
// 读取完整的 float32 向量

void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt);
// 读取向量的分片：[start : start+cnt)
}
```

## 4. Usage Example

```cpp
// Example: Using TinyShakespeareDataset and Tokenizer for Text Generation

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "infini_train/include/nn/modules/gpt2.h"

// 1. 加载数据集（用于训练）
size_t sequence_length = 256;
TinyShakespeareDataset dataset("data/tinyshakespeare/train.bin", sequence_length);

// 获取一个训练样本
auto [x, y] = dataset[0];
// x 形状: (256,), 包含 tokens [t0, t1, ..., t255]
// y 形状: (256,), 包含 tokens [t1, t2, ..., t256]

std::cout << "Dataset size: " << dataset.Size() << " samples\n";

// 2. 加载分词器（用于推理）
infini_train::Tokenizer tokenizer("data/tokenizer/gpt2.tokenizer");

// 解码单个 token
std::string text = tokenizer.Decode(464); // "The"
std::cout << "Decoded text: " << text << "\n";

// 3. 使用预训练模型生成文本
infini_train::nn::GPT2 gpt2(vocab_size, embed_dim, num_layers, num_heads);
gpt2.LoadFromFile("models/gpt2.bin");

// 在 GPU 上生成 100 个 tokens
auto* cuda_device = infini_train::DeviceManager::Instance()->GetDevice("cuda:0");
tokenizer.GenerateText(gpt2, /*batch_size=*/1, /*sequence_length=*/64,
                       /*text_length=*/100, cuda_device);
// 输出: "The meaning of life is ..."（根据模型生成）

// 4. 使用工具函数读取模型权重（自定义加载）
std::ifstream model_file("models/custom.bin", std::ios::binary);
float* weight_matrix = new float[rows * cols];
infini_train::ReadMatrixRowShardFloat(model_file, weight_matrix, rows, cols,
                                      /*row_start=*/0, /*row_cnt=*/rows);
```

## 5. Implementation Details

### Binary File Format Handling
- **Tiny Shakespeare Dataset Format**: 自定义二进制格式，由 1024 字节头部和数据区组成
  - 头部布局：magic (4 bytes) | version (4 bytes) | num_tokens (4 bytes) | reserved (1012 bytes)
  - 数据区：根据 magic number 决定使用 UINT16（GPT-2）或 UINT32（LLaMA 3）编码
  - 所有数据在加载时统一转换为 INT64 类型，方便后续处理
  - 构造函数验证序列长度限制（GPT-2: ≤1024, LLaMA-3: ≤8192）

- **Tokenizer File Format**: 自定义词汇表存储格式
  - 头部：magic (4 bytes) | version (4 bytes) | vocab_size (4 bytes) | eot_token (4 bytes, V2 only)
  - 数据区：每个 token 对应一个变长字符串，格式为 `[length (1 byte)][string data]`
  - 支持版本 1（EOT from lookup table）和版本 2（EOT from header）

### Memory Management
- **Zero-Copy Views**: `TinyShakespeareDataset::operator[]` 使用 `Tensor` 的偏移量构造函数，创建指向原始数据的视图，避免复制。每个样本的字节偏移量通过 `idx * sequence_size_in_bytes_` 计算
- **一次性加载策略**: 数据集和分词器均在构造时将整个文件读入内存，适合中小型数据集（Tiny Shakespeare 约 1MB）
- **类型提升转换**: 读取 UINT16/UINT32 token 数据后自动转换为 INT64，在 `ReadTinyShakespeareFile` 中通过 `static_cast<int64_t>` 实现

### Concurrency
- **无锁设计**: 当前实现为单线程，所有文件 I/O 和数据转换操作均串行执行
- **状态共享**: `GenerateText` 中的随机数生成器使用局部状态（`uint64_t kRngState`），避免全局可变状态

### Performance
- **预分配缓冲区**: 文件读取时预先分配 `std::vector<uint8_t>` 或 `std::vector<T>` 缓冲区，减少动态内存分配
- **分块 I/O 优化**: `ReadMatrixRowShardFloat` 和 `ReadMatrixColShardFloat` 使用 `std::streamoff` 直接定位到目标位置，避免读取无关数据
  - 行分片：单次 `seekg` + 单次 `read`
  - 列分片：逐行 `seekg` + 读取（由于 row-major 布局必须逐行跳跃）
- **编译期类型检查**: `BytesToType` 使用 `static_assert(std::is_trivially_copyable<T>::value)` 确保仅对平凡可复制类型进行反序列化

### Error Handling
- **致命错误**: 使用 `LOG(FATAL)`（来自 glog）在文件不存在、魔数不匹配、版本不支持时终止程序
- **断言检查**: 使用 `CHECK_EQ`、`CHECK_LE` 验证数据类型、维度匹配、序列长度限制
- **边界保护**: `Tokenizer::Decode` 对越界 token ID 返回占位符 "[INVALID_TOKEN]"，而非崩溃

### Dependencies
- **外部库**:
  - `glog`: 用于日志记录（`LOG(INFO)`, `LOG(FATAL)`）
  - `std::filesystem`: 检查文件存在性（C++17）
- **内部依赖**:
  - `infini_train/include/dataset.h`: `Dataset` 基类接口
  - `infini_train/include/tensor.h`: `Tensor` 类，用于数据存储和设备传输
  - `infini_train/include/device.h`: `Device` 和 `DeviceManager`，用于设备管理
  - `infini_train/include/nn/functional.h`: `Softmax` 函数
  - `infini_train/include/nn/modules/module.h`: `Module` 基类

### Design Patterns
- **RAII**: 所有文件句柄（`std::ifstream`）在函数作用域内自动管理，无手动 close 调用
- **CRTP (Curiously Recurring Template Pattern)**: `Dataset` 基类使用虚函数接口，`TinyShakespeareDataset` 实现多态
- **Variant 类型安全**: `ReadTinyShakespeareFile` 使用 `std::variant<std::vector<uint16_t>, std::vector<int32_t>>` 存储不同编码类型的缓冲区，通过 `std::visit` 进行类型安全访问
- **静态映射表**: `kTypeMap`, `kEotMap`, `kPromptMap` 等使用 `const std::unordered_map` 在编译期初始化，提供 O(1) 查找

### Algorithmic Details
- **Multinomial Sampling**: `SampleMult` 实现基于 CDF 的多项式采样，时间复杂度 O(n)，n 为词汇表大小。对于大型词汇表（如 LLaMA-3 的 128K），可能需要优化（如 Alias Method）
- **Xorshift* PRNG**: `RandomU32` 使用 Xorshift* 算法，周期为 2^64 - 1，通过乘以特定常数（0x2545F4914F6CDD1D）提升输出均匀性
- **Softmax + Temperature**: 当前实现未应用温度参数（temperature），直接使用 Softmax 概率进行采样。可通过修改 `SampleMult` 前的 logits 实现温度控制

### Device Abstraction
- **异构计算支持**: `GenerateText` 方法接受 `Device*` 参数，支持在 CPU、CUDA 等不同设备上执行模型推理
- **数据传输**: 使用 `Tensor::To(device)` 在设备间传输数据，生成过程中需要在计算设备和 CPU 之间往返（CPU 存储序列，计算设备执行 Forward）
