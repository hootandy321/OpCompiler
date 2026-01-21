# DataLoader 模块核心实现文档

DataLoader 模块负责在多设备分布式环境下高效加载和管理大语言模型的权重参数。该模块实现了基于行（ROW）、列（COLUMN）和完整复制（FULL）的三种权重分布策略，支持跨多个 GPU 设备的并行权重加载和同步。

## 1. 模块结构

- **`weights_loader.hpp`**: 定义权重加载器的核心接口，包括 `Weight` 类和 `Loader` 类，以及权重分布类型的枚举定义。
- **`weights_loader.cpp`**: 实现权重加载的核心逻辑，包括分布式数据重排、内存拷贝、多设备流同步等底层操作。

## 2. 核心类

### `Weight`
- **位置**: `weights_loader.hpp`
- **主要功能**: 封装单个权重张量的元数据和加载行为，支持在分布式训练/推理场景下按不同策略切分和加载权重。
- **关键成员**:
  - `_tensor`: `std::shared_ptr<Tensor>` - 实际的权重张量对象，使用共享指针管理生命周期
  - `_rank`: `int` - 当前权重所属的 rank 编号（用于标识分布式环境中的设备编号）
  - `_nrank`: `int` - 总 rank 数量（即总设备数），用于计算切分偏移量
  - `_dist_type`: `DistributionType` - 权重分布类型（FULL/ROW/COLUMN）
- **核心方法**:
  - `load(const void *host_data, infinirtStream_t stream)`: 将主机端数据加载到设备端张量中
    - 对于 **FULL** 模式：直接加载完整数据
    - 对于 **ROW** 模式或一维张量：根据 `_rank` 计算字节偏移量（`_rank * _tensor->numel() * dsize(_tensor->dtype())`），从对应位置开始连续加载
    - 对于 **COLUMN** 模式且维度 > 1：执行跨步数据重排，逐行从主机端交错读取列分片数据到连续缓冲区，然后加载到张量
    - 算法复杂度：COLUMN 模式为 O(rows)，其中 rows 是张量的总行数
- **生命周期**: 由 `Loader` 类通过 `register_weight` 方法创建并持有，生命周期与 `Loader` 绑定。

### `Loader`
- **位置**: `weights_loader.hpp` / `weights_loader.cpp`
- **主要功能**: 管理多个设备的权重加载过程，为每个设备创建独立的 CUDA 流和权重映射表，提供注册、加载、查询和最终清理的完整生命周期管理。
- **关键成员**:
  - `_weights_maps`: `std::vector<std::unordered_map<std::string, std::shared_ptr<Weight>>>` - 每个 rank 对应一个权重哈希表，键为权重名称，值为 Weight 对象
  - `_device`: `infiniDevice_t` - 设备类型（如 CUDA、Ascend 等）
  - `_dev_ids`: `std::vector<int>` - 物理设备 ID 列表
  - `_streams`: `std::vector<infinirtStream_t>` - 每个设备对应的异步流，用于并行加载
- **核心方法**:
  - `Loader(infiniDevice_t dev, const std::vector<int> &dev_ids)`: 构造函数
    - 为每个设备调用 `infinirtSetDevice` 设置当前设备
    - 为每个设备创建独立的 CUDA 流（`infinirtStreamCreate`）
    - 初始化每个 rank 的权重哈希表
    - 时间复杂度：O(n)，n 为设备数量

  - `register_weight(const std::string &name, std::shared_ptr<Tensor> tensor, int rank, DistributionType dist_type)`: 注册权重到指定 rank
    - 在 `_weights_maps[rank]` 中插入名称到 Weight 对象的映射
    - 将 `_dev_ids.size()` 作为 `_nrank` 传入 Weight 构造函数
    - 平均时间复杂度：O(1)

  - `load(const std::string &name, const void *host_data)`: 并行加载权重到所有设备
    - **阶段 1 - 异步加载**：遍历所有 rank，设置当前设备（`infinirtSetDevice`），从 `_weights_maps[rank]` 查找权重，调用 `Weight::load` 在各自流上异步执行数据传输
    - **阶段 2 - 同步等待**：逆序遍历所有 rank（从末尾到 0），调用 `infinirtStreamSynchronize` 确保所有流完成操作
    - 同步策略：使用逆序同步避免潜在的资源竞争问题
    - 查找失败时输出错误信息并调用 `std::abort()` 终止程序
    - 时间复杂度：O(n * m)，n 为设备数量，m 为权重张量大小（取决于分布类型）

  - `finalize()`: 清理所有设备的流资源
    - 保存当前设备 ID（`infinirtGetDevice`）
    - 遍历所有设备，同步流（`infinirtStreamSynchronize`），销毁流（`infinirtStreamDestroy`）
    - 恢复原始设备 ID
    - 防止资源泄漏的必要步骤

  - `get(const std::string &name, int rank)`: 获取指定 rank 的权重张量
    - 直接返回 `_weights_maps[rank][name]` 的底层 Tensor 对象
    - 平均时间复杂度：O(1)

- **生命周期**: 由外部创建并持有，应在所有权重加载完成后调用 `finalize()` 释放资源。

## 3. API 接口

```cpp
// 权重分布类型枚举
enum DistributionType {
    FULL,    // 完整复制：每个设备保存完整权重副本
    ROW,     // 行分布：按行切分权重（适用于 Embedding、LayerNorm 等）
    COLUMN   // 列分布：按列切分权重（适用于 Linear 层权重矩阵）
};

// 加载单个权重到设备
void Weight::load(const void *host_data, infinirtStream_t stream = nullptr);
// host_data: 主机端源数据指针
// stream: CUDA 流句柄，支持异步加载

// 注册权重张量到加载器
void Loader::register_weight(const std::string &name,
                             std::shared_ptr<Tensor> tensor,
                             int rank = 0,
                             DistributionType dist_type = DistributionType::FULL);
// name: 权重唯一标识符
// tensor: 目标设备端张量
// rank: 目标 rank 编号
// dist_type: 分布策略

// 并行加载权重到所有设备
void Loader::load(const std::string &name, const void *host_data);
// name: 已注册的权重名称
// host_data: 主机端完整权重数据

// 获取指定 rank 的权重张量
std::shared_ptr<Tensor> Loader::get(const std::string &name, int rank = 0);

// 清理资源
void Loader::finalize();
```

## 4. 使用示例

```cpp
// 示例：在 4 卡 GPU 环境下加载分布式 LLaMA 模型权重
#include "dataloader/weights_loader.hpp"

using namespace infinicore;
using namespace infinicore::weights;

// 1. 创建加载器，指定设备类型和 GPU ID 列表
std::vector<int> gpu_ids = {0, 1, 2, 3};
Loader loader(INFINI_DEVICE_CUDA, gpu_ids);

// 2. 为每个 rank 注册权重张量
// 假设模型有 embedding 层（行分布）和 transformer 层（列分布）
for (int rank = 0; rank < 4; rank++) {
    // Embedding 层：按行切分（每个 GPU 存储部分 token）
    auto embed_tensor = std::make_shared<Tensor>(...); // 创建设备端张量
    loader.register_weight("model.embed_tokens.weight", embed_tensor,
                          rank, DistributionType::ROW);

    // Transformer 层权重：按列切分（每个 GPU 存储部分输出维度）
    auto q_proj_tensor = std::make_shared<Tensor>(...);
    loader.register_weight("layers.0.self_attn.q_proj.weight", q_proj_tensor,
                          rank, DistributionType::COLUMN);

    // LayerNorm 层：完整复制（每个 GPU 存储完整权重）
    auto ln_tensor = std::make_shared<Tensor>(...);
    loader.register_weight("layers.0.input_layernorm.weight", ln_tensor,
                          rank, DistributionType::FULL);
}

// 3. 从磁盘读取权重数据并加载
// 假设 host_weights 是从 safetensors 文件读取的完整权重数据
void* host_weights = load_weights_from_file("llama-7b.safetensors");

// 并行加载到所有 GPU（内部自动处理切分逻辑）
loader.load("model.embed_tokens.weight", host_weights);
loader.load("layers.0.self_attn.q_proj.weight", host_weights);
loader.load("layers.0.input_layernorm.weight", host_weights);

// 4. 在计算图中使用权重
int my_rank = 0; // 当前进程的 rank
auto q_proj_weight = loader.get("layers.0.self_attn.q_proj.weight", my_rank);
// 使用 q_proj_weight 进行矩阵乘法等操作

// 5. 清理资源
loader.finalize(); // 销毁所有 CUDA 流
```

## 5. 实现细节

### 内存管理
- **设备内存分配**: 通过底层 Tensor 类管理设备端内存（具体分配策略由 Tensor 类实现）
- **主机端临时缓冲**: COLUMN 分布模式下使用 `infinirtMallocHost` 分配页锁定主机内存（pinned memory），用于数据重排，提升 PCIe 传输效率
- **自动释放**: 使用 `std::shared_ptr` 管理 Weight 对象生命周期，`finalize()` 中显式释放 CUDA 流资源

### 并发控制
- **设备隔离**: 每次加载前调用 `infinirtSetDevice` 确保操作在正确设备上执行
- **流并行**: 每个 GPU 拥有独立的 CUDA 流（`_streams[rank]`），实现多设备数据传输并行化
- **同步屏障**: 加载完成后逆序同步所有流（`infinirtStreamSynchronize`），确保所有设备的传输完成后再返回
- **无锁设计**: 使用 per-rank 哈希表，避免跨设备共享状态的锁竞争

### 性能优化
- **异步传输**: 使用 CUDA 流实现主机到设备的异步内存拷贝，掩盖传输延迟
- **零拷贝优化**: ROW 分布模式下直接计算偏移量，从主机端数据的对应位置连续拷贝，无需中间缓冲
- **预分配内存**: Tensor 对象在注册时已分配设备内存，加载阶段仅执行数据传输
- **页锁定内存**: COLUMN 重排时使用 `infinirtMallocHost` 分配 pinned memory，提升 DMA 传输速度
- **批量同步**: 所有设备的流在加载完成后统一同步，减少同步开销

### 错误处理
- **查找失败**: `load()` 方法中若权重未注册会输出错误信息并调用 `std::abort()` 终止程序
- **内存分配失败**: `RUN_INFINI` 宏包装 infinirt API 调用，分配失败时抛出异常
- **不支持分布类型**: `Weight::load` 中若遇到未定义的 `_dist_type` 会输出错误并终止
- **设备恢复**: `finalize()` 中保存并恢复原始设备 ID，避免状态污染

### 依赖关系
- **外部依赖**:
  - `infinirt.h`: 抽象的运行时 API，提供跨平台设备管理、内存分配、流操作接口
  - `../tensor.hpp`: Tensor 类，封装设备端张量操作（`load()`, `ndim()`, `shape()`, `dtype()`, `numel()`）
  - `../utils.hpp`: 工具函数（如 `dsize()` 获取数据类型字节数）
  - `infinicore_infer/weights_loader.h`: C FFI 接口定义
- **数据流**: 磁盘文件 → 主机端完整权重 → (可选重排) → 设备端分片权重

### 设计模式
- **工厂模式**: `Loader` 作为 Weight 对象的工厂，通过 `register_weight` 创建实例
- **策略模式**: `DistributionType` 枚举定义不同的加载策略，`Weight::load` 根据类型选择算法
- **RAII**: 使用 shared_ptr 自动管理 Tensor 和 Weight 对象生命周期，`Loader` 析构时调用 `finalize`
- **外观模式**: `Loader` 提供简化的高层接口，隐藏多设备、流同步等底层复杂性
- **C FFI**: 提供 `loadModelWeight` C 接口，支持跨语言调用（如 Python 绑定）

### 算法细节
- **COLUMN 重排算法**（`weights_loader.cpp:16-30`）:
  - 输入：交错存储的完整主机端矩阵（列主序分片）
  - 输出：连续存储的列分片数据
  - 步骤：
    1. 计算单行字节数：`row_size = shape[-1] * dsize(dtype)`
    2. 计算当前 rank 的列偏移：`host_offset = rank * row_size`
    3. 计算主机端行跨距：`host_row_size = nrank * row_size`
    4. 遍历每一行，从交错位置拷贝 `row_size` 字节到连续缓冲区
  - 示例：4 个 rank，3 行 8 列矩阵，rank 1 获取第 1, 5, 9 列数据
- **分布策略选择**:
  - FULL: LayerNorm、RMSNorm 等归一化层参数，以及维度较小的 bias
  - ROW: Embedding 层（按 token 维度切分）、输出层（按 vocab 维度切分）
  - COLUMN: Linear 层权重矩阵（按输出维度切分，实现模型并行）
