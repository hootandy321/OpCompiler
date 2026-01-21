# Neural Network Modules Core Implementation Documentation

本模块实现了 InfiniTrain 框架的核心神经网络层组件，提供类似 PyTorch 的模块化 API，支持前向传播、参数管理、设备迁移和数据并行复制。

## 1. Module Structure

- **`module.cc`**: 核心 Module 基类实现，提供参数管理、设备迁移、状态字典和子模块遍历功能
- **`linear.cc`**: 全连接层 (Linear/Dense) 实现，支持可偏置的仿射变换
- **`activations.cc`**: 激活函数层 (当前实现 Sigmoid)
- **`normalization.cc`**: 层归一化 (LayerNorm) 实现
- **`sparse.cc`**: 稀疏层 (Embedding) 实现，用于词嵌入
- **`loss.cc`**: 损失函数层 (CrossEntropyLoss) 实现
- **`container.cc`**: 容器模块 (Sequential, ModuleDict, ModuleList) 实现

## 2. Core Classes

### `Module`
- **Location**: `module.cc` / `include/nn/modules/module.h`
- **Primary Function**: 所有神经网络模块的抽象基类，提供参数管理、设备管理和模块组合的核心基础设施
- **Key Members**:
  - `device_`: `const Device*` - 模块所在的设备指针，默认使用 DeviceManager::Instance()->GetDefaultDevice()
  - `type_`: `std::string` - 模块类型标识符
  - `modules_`: `std::unordered_map<std::string, std::shared_ptr<Module>>` - 子模块的有序映射表
  - `parameters_`: `std::unordered_map<std::string, std::shared_ptr<Tensor>>` - 可学习参数的映射表
  - `buffers_`: `std::unordered_map<std::string, std::shared_ptr<Tensor>>` - 不可学习状态张量（如 BatchNorm 的 running_mean）
- **Core Methods**:
  - `Parameters()`: 递归收集本模块及所有子模块的参数，使用 `std::unordered_set<const Tensor*> visited` 防止重复收集，返回扁平化的参数列表，时间复杂度 O(N) 其中 N 为子模块总数
  - `Buffers()`: 递归收集本模块及所有子模块的 buffer 张量，不进行去重处理
  - `StateDict()`: 构建完整的状态字典，包含所有参数和 buffers，跳过名称以 "__pp" 开头的子模块（用于 Pipeline Parallel），返回带层级前缀的扁平化映射（如 "layer1.weight"）
  - `To(const Device*)`: 将模块迁移到目标设备，使用 `Tensor::To(device)` 创建新的参数和 buffers 副本，然后递归迁移所有子模块，通过移动语义交换 containers 以保证原子性
  - `To(DataType)`: 将模块的所有参数和 buffers 转换为目标数据类型
  - `Apply(std::function<void(Module*)>)`: 深度优先遍历所有子模块并应用函数，最后对自身调用该函数
  - `NamedModules(prefix, remove_duplicate, memory)`: 递归遍历模块树生成带名称前缀的模块映射，使用 `std::unordered_set<Module*> memory` 检测循环引用并去重，通过 `remove_duplicate` 参数控制是否跳过重复模块
  - `modules()`: 返回所有模块的扁平化列表（包括自身），使用 NamedModules 实现，过滤掉空字符串名称的根模块
- **Lifecycle**: 继承 `std::enable_shared_from_this<Module>` 以支持在成员函数中获取 `shared_ptr`，支持默认构造（type 为 "Undefined"）和类型指定构造，提供虚析构函数，支持值拷贝构造（默认）

### `CloneableModule<Derived>`
- **Location**: `include/nn/modules/module.h`
- **Primary Function**: CRTP (Curiously Recurring Template Pattern) 模板基类，为派生类提供类型安全的复制功能
- **Key Members**: 无额外成员，完全通过 Derived 类型参数化
- **Core Methods**:
  - `ReplicateForDataParallel(int device_idx)`: 重写基类虚函数，返回 `std::make_shared<Derived>(static_cast<const Derived&>(*this))`，实现派生类的类型安全深拷贝
- **Design Pattern**: CRTP (Static Polymorphism)，避免虚函数开销，支持编译时类型推导

### `Linear`
- **Location**: `linear.cc` / `include/nn/modules/linear.h`
- **Primary Function**: 实现仿射变换 `y = xA^T + b`，支持可选偏置项
- **Key Members**:
  - `bias_`: `bool` - 是否使用偏置项的标志
  - `parameters_["weight"]`: `Tensor[out_features, in_features]` - 权重矩阵，使用 Kaiming Uniform 初始化
  - `parameters_["bias"]`: `Tensor[out_features]` - 偏置向量（可选），使用 Uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)] 初始化
- **Core Methods**:
  - `Forward(input_tensors)`: 调用 `autograd::Linear` 实现可微分的前向传播，根据 `bias_` 标志动态构建输入张量列表（{input, weight} 或 {input, weight, bias}）
  - `ResetParameters()`: 权重使用 `init::KaimingUniform(sqrt(5))` 初始化（He 初始化），偏置使用 `init::Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` 初始化
- **Initialization Strategy**: Kaiming Uniform (He et al., 2015) 适用于 ReLU 激活函数，增益系数为 sqrt(5)
- **Memory Layout**: 权重矩阵形状为 `[out_features, in_features]`，支持高效矩阵乘法

### `Sigmoid`
- **Location**: `activations.cc` / `include/nn/modules/activations.h`
- **Primary Function**: 实现 Sigmoid 激活函数 `sigmoid(x) = 1 / (1 + e^{-x})`
- **Key Members**: 无参数或 buffers
- **Core Methods**:
  - `Forward(input_tensors)`: 委托给 `autograd::Sigmoid->Apply(input_tensors)` 实现可微分的激活函数计算
- **Design Pattern**: 适配器模式，将 autograd 操作包装为 Module 接口

### `LayerNorm`
- **Location**: `normalization.cc` / `include/nn/modules/normalization.h`
- **Primary Function**: 实现层归一化 `y = (x - mean) / sqrt(var + eps) * gamma + beta`，在特征维度上归一化
- **Key Members**:
  - `eps_`: `float` - 防止除零的小常数，默认 1e-5
  - `parameters_["weight"]`: `Tensor[normalized_shape]` - 可学习的缩放参数 gamma，初始化为 1
  - `parameters_["bias"]`: `Tensor[normalized_shape]` - 可学习的偏移参数 beta，初始化为 0
- **Core Methods**:
  - `Forward(input_tensors)`: 调用 `autograd::LayerNorm(eps_)->Apply({input, weight, bias})` 实现可微分的层归一化
  - `ResetParameters()`: 使用 `init::Ones(weight)` 和 `init::Zeros(bias)` 初始化参数
- **Normalization Strategy**: 在最后 D 个维度上计算均值和方差，支持任意维度的输入张量

### `Embedding`
- **Location**: `sparse.cc` / `include/nn/modules/sparse.h`
- **Primary Function**: 实现离散索引到连续向量的查找表，用于词嵌入和类别特征
- **Key Members**:
  - `parameters_["weight"]`: `Tensor[num_embeddings, embedding_dim]` - 嵌入矩阵，使用正态分布初始化
- **Core Methods**:
  - `Forward(input_tensors)`: 接收索引张量 `indices`，调用 `autograd::Embedding->Apply({indices, weight})` 实现可微分的嵌入查找
  - `ResetParameters()`: 使用 `init::Normal(weight)` 初始化嵌入矩阵
- **Sparse Operation**: 使用 gather 操作从权重矩阵中提取指定索引的行，梯度更新时只修改被访问的行

### `CrossEntropyLoss`
- **Location**: `loss.cc` / `include/nn/modules/loss.h`
- **Primary Function**: 实现交叉熵损失 `H(p, q) = -sum(p_i * log(q_i))`，用于分类任务
- **Key Members**: 无参数
- **Core Methods**:
  - `Forward(input_tensors)`: 委托给 `autograd::CrossEntropy->Apply(input_tensors)`，期望输入为 {logits, targets}
- **Numerical Stability**: 在 autograd 层实现 log_softmax 与 nll_loss 的融合，避免数值下溢

### `Sequential`
- **Location**: `container.cc` / `include/nn/modules/container.h`
- **Primary Function**: 顺序容器，按顺序执行子模块，前一个模块的输出是后一个模块的输入
- **Key Members**:
  - `modules_`: 从父类继承，存储字符串索引到子模块的映射，索引为 "0", "1", "2", ...
- **Core Methods**:
  - `Sequential(std::vector<std::shared_ptr<Module>>&& layers)`: 构造函数使用移动语义接收模块列表，按顺序将每个模块存入 `modules_[std::to_string(idx)]`
  - `Forward(input_tensors)`: 迭代执行所有子模块，每次循环更新 `x = modules_[std::to_string(idx)]->Forward(x)`，通过 `const_cast` 移除输入的 const 限制以支持就地修改
- **Use Case**: 构建深度神经网络，如 `Sequential(Linear, ReLU, Linear, Softmax)`

### `ModuleList`
- **Location**: `container.cc` / `include/nn/modules/container.h`
- **Primary Function**: 可迭代模块列表，支持索引访问，但不在 Forward 中自动执行
- **Key Members**:
  - `module_list_`: `std::vector<std::shared_ptr<Module>>` - 模块的向量存储
  - `modules_`: 从父类继承，为支持 NamedModules 而维护的索引映射
- **Core Methods**:
  - `ModuleList(std::vector<std::shared_ptr<Module>>&& layers)`: 构造函数同时填充 `module_list_` 和 `modules_`
  - `Forward(input_tensors)`: 未实现，抛出 `LOG(FATAL) << "Not implemented"`
  - `begin()`, `end()`: 提供 C++ 风格迭代器支持 range-based for 循环
  - `operator[](std::size_t idx)`: 返回指定索引的模块引用
- **Use Case**: 存储 Transformer 的多个 encoder 层，手动控制执行流程

### `ModuleDict`
- **Location**: `container.cc` / `include/nn/modules/container.h`
- **Primary Function**: 字典容器，通过字符串键访问子模块
- **Key Members**:
  - `modules_`: 从父类继承，存储用户指定的键值对
- **Core Methods**:
  - `ModuleDict(std::unordered_map<std::string, std::shared_ptr<Module>> modules)`: 构造函数移动插入所有键值对
  - `Forward(input_tensors)`: 未实现，抛出 `LOG(FATAL) << "Not implemented"`
- **Use Case**: 多任务学习或条件路由的模块选择

## 3. API Interface

```cpp
// Module 基类核心接口
class Module {
public:
    explicit Module();
    explicit Module(const std::string &type);

    // 参数管理
    virtual std::vector<std::shared_ptr<Tensor>> Parameters() const;
    bool has_parameter(const std::string &name) const;
    std::shared_ptr<Tensor> *mutable_parameter(const std::string &name);
    const std::shared_ptr<Tensor> &parameter(const std::string &name) const;

    // Buffer 管理
    virtual std::vector<std::shared_ptr<Tensor>> Buffers() const;

    // 子模块访问
    std::vector<std::shared_ptr<Module>> modules();
    std::shared_ptr<Module> mutable_module(const std::string &name);
    const Module &module(const std::string &name) const;

    // 状态管理
    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const;

    // 前向传播（需子类实现）
    virtual std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors);

    // 设备和数据类型迁移
    virtual void To(const Device *device);
    virtual void To(DataType dtype);

    // 函数式遍历
    void Apply(std::function<void(Module *)> fn);

    // 数据并行复制
    virtual std::shared_ptr<Module> ReplicateForDataParallel(int device_idx) const;
};

// Linear 层接口
class Linear : public CloneableModule<Linear> {
public:
    Linear(int64_t in_features, int64_t out_features,
           bool bias = true, const Device *device = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

// Sequential 容器接口
class Sequential : public CloneableModule<Sequential> {
public:
    explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

// LayerNorm 接口
class LayerNorm : public CloneableModule<LayerNorm> {
public:
    LayerNorm(const std::vector<int64_t> &normalized_shape,
              float eps = 1e-5f, const Device *device = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

// Embedding 接口
class Embedding : public CloneableModule<Embedding> {
public:
    Embedding(int num_embeddings, int embedding_dim,
              const Device *device = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
```

## 4. Usage Example

```cpp
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"

using namespace infini_train::nn;

// 构建 3 层 MLP 模型
auto model = std::make_shared<Sequential>(
    std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(784, 512),   // 输入层: 784 -> 512
        std::make_shared<Sigmoid>(),            // 激活函数
        std::make_shared<Linear>(512, 256),    // 隐藏层: 512 -> 256
        std::make_shared<Sigmoid>(),            // 激活函数
        std::make_shared<Linear>(256, 10)      // 输出层: 256 -> 10
    }
);

// 创建词嵌入层
auto embedding = std::make_shared<Embedding>(
    10000,  // vocab_size: 10000 个 token
    512,    // embedding_dim: 每个token 映射为 512 维向量
    device  // 目标设备
);

// 创建层归一化
auto layer_norm = std::make_shared<LayerNorm>(
    std::vector<int64_t>{512},  // 在最后 512 个维度上归一化
    1e-5f,                       // eps 参数
    device
);

// 前向传播
auto input_tensor = std::make_shared<Tensor>(...);
auto output_tensors = model->Forward({input_tensor});

// 访问模型参数
auto all_params = model->Parameters();  // 获取所有可学习参数
auto weight = model->modules()[0]->parameter("weight");  // 获取第一层权重

// 设备迁移
auto cuda_device = DeviceManager::Instance()->GetDevice("cuda:0");
model->To(cuda_device);  // 将整个模型迁移到 GPU

// 数据类型转换
model->To(DataType::kFLOAT16);  // 转换为半精度浮点数

// 保存模型状态
auto state_dict = model->StateDict();
// state_dict["0.weight"], state_dict["0.bias"], state_dict["2.weight"], ...

// 应用函数到所有模块
model->Apply([](Module* module) {
    if (module->type() == "Linear") {
        // 对所有 Linear 层执行自定义操作
    }
});

// ModuleList 使用示例
auto layers = std::make_shared<ModuleList>(
    std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(256, 256),
        std::make_shared<LayerNorm>(std::vector<int64_t>{256}),
        std::make_shared<Linear>(256, 256)
    }
);

// 手动迭代执行
auto x = input_tensor;
for (auto& layer : *layers) {
    x = layer->Forward({x})[0];
}
```

## 5. Implementation Details

### Memory Management
- **共享所有权**: 所有模块和参数使用 `std::shared_ptr` 管理，支持引用计数和自动生命周期管理
- **enable_shared_from_this**: Module 继承 `std::enable_shared_from_this<Module>`，允许在成员函数中安全地获取 `shared_ptr`，避免悬空指针
- **移动语义**: Sequential 构造函数使用 `std::vector<std::shared_ptr<Module>>&&` 右值引用，避免不必要的引用计数操作
- **参数去重**: `Parameters()` 方法使用 `std::unordered_set<const Tensor*> visited` 检测并跳过共享的参数指针，防止重复收集

### Concurrency
- **线程不安全**: 当前实现未使用任何互斥锁或原子操作，不支持多线程并发修改模块状态
- **只读并发**: StateDict(), Parameters(), Buffers() 等只读方法在单次调用内是线程安全的（不修改内部状态），但与其他写操作并发时未定义行为
- **设备迁移**: `To()` 方法在迁移过程中创建新的 Tensor 副本，通过移动语义交换 containers，保证操作的原子性，避免部分迁移状态

### Performance
- **递归遍历**: `Parameters()`, `StateDict()`, `Apply()` 等方法使用深度优先遍历，时间复杂度 O(N)，空间复杂度 O(H) 其中 H 为模块树的深度
- **哈希表查找**: `parameters_`, `modules_`, `buffers_` 使用 `std::unordered_map`，提供平均 O(1) 的查找时间
- **避免虚函数开销**: CloneableModule 使用 CRTP 静态多态，`ReplicateForDataParallel()` 在编译时解析为派生类型的构造函数调用
- **内联优化**: 简单方法（如 `type()`, `has_parameter()`）在头文件中定义，允许编译器内联优化

### Error Handling
- **运行时断言**: 使用 `CHECK()` (Google Logging) 进行参数验证，如 `CHECK(parameters_.find(name) != parameters_.end())`
- **未实现虚函数**: 基类 `Forward()` 抛出 `LOG(FATAL) << "Forward function not implemented for this module"`
- **空指针检查**: `To()` 方法使用 `CHECK_NOTNULL(device)` 验证设备指针有效性
- **状态一致性**: `NamedModules()` 使用 `CHECK(!named_modules.contains(prefix))` 确保不覆盖已注册的模块

### Dependencies
- **autograd 模块**: 所有具体层 (Linear, Sigmoid, LayerNorm, Embedding, CrossEntropyLoss) 依赖 `autograd::` 命名空间下的对应函数类实现前向传播和自动微分
- **Tensor 核心**: 依赖 `Tensor` 类的 `To(device)`, `To(dtype)`, `RequiresGrad()` 方法
- **Device 管理**: 依赖 `DeviceManager::Instance()->GetDefaultDevice()` 获取默认设备
- **初始化模块**: 依赖 `init::KaimingUniform`, `init::Uniform`, `init::Normal`, `init::Ones`, `init::Zeros`, `init::CalculateFanInAndFanOut` 进行参数初始化
- **日志系统**: 使用 `glog/logging.h` 提供的 `LOG()`, `CHECK()` 宏进行日志记录和断言

### Design Patterns
- **Composite Pattern**: Module 作为树形结构的基类，支持任意深度的模块嵌套（如 Sequential 中包含 Sequential）
- **Template Method Pattern**: `Forward()` 定义算法骨架，具体计算委托给 `autograd::` 子系统
- **Adapter Pattern**: nn::modules 类将 autograd 函数包装为 Module 接口，解耦自动微分与模型构建
- **CRTP (Curiously Recurring Template Pattern)**: CloneableModule 使用静态多态实现类型安全的复制，避免虚函数开销
- **Iterator Pattern**: ModuleList 提供 `begin()`, `end()` 迭代器，支持 range-based for 循环和 STL 算法
- **Factory Method Pattern**: `ReplicateForDataParallel()` 作为克隆方法，由派生类实现具体复制逻辑

### Pipeline Parallel Support
- **特殊命名约定**: `StateDict()` 跳过名称以 `"__pp"` 开头的子模块，用于 Pipeline Parallel 阶段间边界
- **预定义常量**:
  - `kPPFirstStageName`: `"__pp_first_stage"` - 流水线第一个阶段
  - `kPPLastStageName`: `"__pp_last_stage"` - 流水线最后一个阶段
  - `kPPChunkNamePrefix`: `"__pp_chunk_"` - 流水线微批次前缀

### Initialization Strategies
- **Linear**: 权重使用 Kaiming Uniform (He et al., 2015) 适用于 ReLU 及其变体，公式为 `bound = gain * sqrt(3 / fan_in)`，其中 gain = sqrt(5)
- **LayerNorm**: weight 初始化为 1 (保持原始方差)，bias 初始化为 0 (保持原始均值)
- **Embedding**: 使用正态分布初始化，均值为 0，标准差为 1（可能需要调整为特定任务）
- **初始化时机**: 参数在构造函数中分配内存并初始化，调用 `ResetParameters()` 私有方法

### Known Limitations
- **Forward in Constructor**: `NamedModules()` 注释 "FIXME(dcj): can not call this function in constructor"，可能与 `shared_from_this()` 要求对象已构造有关
- **ModuleDict Forward**: `ModuleDict::Forward()` 抛出 "Not implemented"，未定义如何基于字典键路由输入
- **ModuleList Forward**: `ModuleList::Forward()` 抛出 "Not implemented"，要求用户手动迭代调用子模块
- **设备索引未使用**: `ReplicateForDataParallel(int device_idx)` 注释 "TODO(dcj): use device_idx later"，当前未实际使用设备索引参数
