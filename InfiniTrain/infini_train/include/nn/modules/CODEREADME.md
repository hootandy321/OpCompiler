# Neural Network Modules Core Implementation Documentation

该模块实现了深度学习框架中神经网络层的基础抽象层和通用组件,提供类似 PyTorch 的模块化接口设计,支持参数管理、设备迁移、数据并行和流水线并行等分布式训练能力。

## 1. Module Structure

- **`module.h`**: 核心基类 Module 和 CloneableModule 的定义,实现参数/缓冲区管理、设备抽象、递归应用、状态字典等基础设施
- **`activations.h`**: 激活函数层实现,包含 Sigmoid 激活
- **`container.h`**: 容器模块,提供 Sequential(顺序容器)、ModuleDict(字典容器)、ModuleList(列表容器)三种组合模式
- **`linear.h`**: 全连接线性层实现,支持可选的偏置项
- **`loss.h`**: 损失函数层,包含交叉熵损失 CrossEntropyLoss
- **`normalization.h`**: 归一化层,包含层归一化 LayerNorm
- **`sparse.h`**: 稀疏层,包含词嵌入 Embedding

## 2. Core Classes

### `Module`
- **Location**: `module.h`
- **Primary Function**: 所有神经网络模块的抽象基类,提供统一的参数管理、前向传播接口、设备迁移、分布式训练支持等核心能力
- **Key Members**:
  - `device_`: `const Device*` - 当前模块绑定的计算设备指针(CUDA/CPU/Kunlun等)
  - `type_`: `std::string` - 模块类型标识字符串,默认为 "Undefined"
  - `modules_`: `std::unordered_map<std::string, std::shared_ptr<Module>>` - 子模块注册表,支持命名访问
  - `parameters_`: `std::unordered_map<std::string, std::shared_ptr<Tensor>>` - 可学习参数字典,如权重和偏置
  - `buffers_`: `std::unordered_map<std::string, std::shared_ptr<Tensor>>` - 非梯度缓冲区,如 BatchNorm 的运行统计量
- **Core Methods**:
  - `Parameters()`: 递归收集当前模块及所有子模块的参数,返回扁平化的 Tensor 向量,用于优化器更新
  - `Buffers()`: 递归收集所有缓冲区张量,用于状态保存和迁移
  - `StateDict()`: 返回完整的状态字典,包含所有命名参数,用于模型检查点保存/加载
  - `Forward(input_tensors)`: 纯虚函数,定义前向传播接口,子类必须实现具体计算逻辑
  - `TrainStep(inputs, targets, loss_fn, dtype)`: 虚函数,默认实现返回 0.0f,支持单步训练钩子
  - `To(device)`: 递归将模块及其所有子模块、参数、缓冲区迁移到指定设备
  - `To(dtype)`: 递归转换所有参数和缓冲区的数据类型(float32/float16/bfloat16)
  - `Apply(fn)`: 深度优先遍历模块树,对每个模块应用函数对象,支持递归操作如 eval/train 模式切换
  - `ReplicateForDataParallel(device_idx)`: 虚函数,为数据并行创建模块副本,基类返回空指针
  - `NamedModules(prefix, remove_duplicate, memory)`: 私有方法,递归构建所有子模块的扁平化命名映射,支持前缀命名和去重
  - `has_parameter(name)`: 检查指定名称的参数是否存在
  - `mutable_parameter(name)`: 返回参数的可修改指针,用于参数初始化或替换
  - `parameter(name)`: 返回参数的常量引用,用于只读访问
- **Lifecycle**: 继承自 `std::enable_shared_from_this<Module>`,支持从成员函数返回 `shared_ptr`;默认构造函数可指定类型字符串;虚析构函数确保正确的多态销毁;支持拷贝构造(默认实现)

### `CloneableModule<Derived>`
- **Location**: `module.h`
- **Primary Function**: CRTP(Curiously Recurring Template Pattern)模板基类,为派生类提供自动的克隆实现,简化数据并行副本创建
- **Key Members**:
  - 无额外成员,通过模板参数 `Derived` 获取派生类类型信息
- **Core Methods**:
  - `ReplicateForDataParallel(device_idx)`: 重写基类虚函数,通过 `static_cast<const Derived&>` 将当前对象转换为派生类引用,调用拷贝构造函数创建新实例,返回包装为 `shared_ptr<Module>` 的副本
- **Lifecycle**: 无特殊生命周期管理,依赖派生类的拷贝构造语义;典型使用场景是每个 GPU 设备上复制一个独立的模型副本

### `Linear`
- **Location**: `linear.h`
- **Primary Function**: 全连接层(也称为 Dense 或 Affine 层),实现 `y = xA^T + b` 线性变换,其中 A 是权重矩阵,b 是可选偏置向量
- **Key Members**:
  - `bias_`: `bool` - 是否包含偏置项的标志,默认为 true
- **Core Methods**:
  - `Linear(in_features, out_features, bias, device)`: 构造函数,创建形状为 `(out_features, in_features)` 的权重矩阵和形状为 `(out_features,)` 的偏置向量(如果 bias=true),调用 `ResetParameters()` 初始化
  - `Forward(input_tensors)`: 接收输入张量,执行矩阵乘法和偏置加法,返回输出张量;输入形状应为 `(batch_size, *, in_features)`,输出形状为 `(batch_size, *, out_features)`
  - `ResetParameters()`: 私有方法,初始化权重和偏置参数,通常使用 Xavier/Kaiming 初始化或均匀分布
- **Lifecycle**: 继承自 `CloneableModule<Linear>`,支持拷贝构造用于数据并行;参数名称常量 `kParamWeightName="weight"` 和 `kParamBiasName="bias"` 用于状态字典序列化

### `Sequential`
- **Location**: `container.h`
- **Primary Function**: 顺序容器模块,按构造顺序依次执行子模块的前向传播,将前一模块的输出作为后一模块的输入
- **Key Members**:
  - 无显式成员变量,子模块存储在基类 `Module::modules_` 中
- **Core Methods**:
  - `Sequential(layers)`: 转发引用构造函数,接收 `std::vector<std::shared_ptr<Module>>&&`,按顺序将模块注册到 `modules_` 映射,键为索引字符串
  - `Forward(input_tensors)`: 遍历子模块列表,依次调用每个模块的 `Forward()`,将上一模块的输出传递给下一模块,最终返回最后一个模块的输出
- **Lifecycle**: 继承自 `CloneableModule<Sequential>`,支持容器拷贝;典型用法如 `Sequential({Linear(...), ReLU(), Linear(...)})`

### `ModuleList`
- **Location**: `container.h`
- **Primary Function**: 有序模块列表容器,支持索引访问和迭代器遍历,`Forward()` 方法默认直接透传输入(不自动调用子模块)
- **Key Members**:
  - `module_list_`: `std::vector<std::shared_ptr<Module>>` - 存储子模块的向量,保持插入顺序
- **Core Methods**:
  - `ModuleList(layers)`: 转发引用构造函数,转移所有权将模块存储到 `module_list_`
  - `Forward(input_tensors)`: 默认实现直接返回输入张量,不自动执行子模块(与 Sequential 不同)
  - `begin()/end()`: 返回迭代器,支持范围 for 循环遍历子模块
  - `operator[](idx)`: 返回指定索引处的模块引用,支持读写访问
- **Lifecycle**: 继承自 `CloneableModule<ModuleList>`,支持列表拷贝;常用于动态层数架构(如 Transformer 的 N 个编码器层)

### `ModuleDict`
- **Location**: `container.h`
- **Primary Function**: 无序字典容器,支持字符串键访问子模块,类似于 Python 的 `OrderedDict`(但当前未保证插入顺序)
- **Key Members**:
  - 无显式成员,子模块存储在基类 `Module::modules_` 中
- **Core Methods**:
  - `ModuleDict(modules)`: 构造函数,接收 `std::unordered_map<std::string, std::shared_ptr<Module>>`,按字符串键注册模块
  - `Forward(input_tensors)`: 默认实现直接返回输入张量,不自动执行子模块
- **Lifecycle**: 继承自 `CloneableModule<ModuleDict>`;适用于多任务学习或专家混合(MoE)架构的模块路由

### `Sigmoid`
- **Location**: `activations.h`
- **Primary Function**: Sigmoid 激活函数层,逐元素应用 `sigmoid(x) = 1 / (1 + e^(-x))` 变换,将输入压缩到 (0, 1) 区间
- **Key Members**:
  - 无成员变量,纯函数式操作
- **Core Methods**:
  - `Forward(input_tensors)`: 接收任意形状的输入张量,逐元素计算 Sigmoid 函数,返回相同形状的输出张量;通常用于二分类的输出层或门控机制
- **Lifecycle**: 继承自 `CloneableModule<Sigmoid>`,无状态可学习参数

### `LayerNorm`
- **Location**: `normalization.h`
- **Primary Function**: 层归一化,对最后一个维度应用归一化 `y = (x - mean) / sqrt(var + eps) * gamma + beta`,用于稳定训练和加速收敛
- **Key Members**:
  - `eps_`: `const float` - 防止除零的小常数,默认为 1e-5
- **Core Methods**:
  - `LayerNorm(normalized_shape, eps, device)`: 构造函数,`normalized_shape` 指定要归一化的维度(如 `[hidden_size]`),创建可学习的 `gamma`(weight)和 `beta`(bias)参数,调用 `ResetParameters()` 初始化
  - `Forward(input_tensors)`: 计算输入张量在指定维度上的均值和方差,执行归一化和仿射变换,返回归一化后的张量
  - `ResetParameters()`: 私有方法,初始化 weight 为 1,bias 为 0
- **Lifecycle**: 继承自 `CloneableModule<LayerNorm>`,参数名称常量为 `kParamWeightName="weight"` 和 `kParamBiasName="bias"`

### `CrossEntropyLoss`
- **Location**: `loss.h`
- **Primary Function**: 交叉熵损失函数,用于分类任务,结合 LogSoftmax 和负对数似然损失 `loss = -log(softmax(x)[target])`
- **Key Members**:
  - 无成员变量
- **Core Methods**:
  - `Forward(input_tensors)`: 接收模型输出 logits 和目标标签,计算交叉熵损失,返回形状为 `(1,)` 的标量损失张量;输入应为 `(batch_size, num_classes)` 的未归一化 logit
- **Lifecycle**: 继承自 `CloneableModule<CrossEntropyLoss>`,无状态参数;典型用法是包装模型输出,计算损失用于反向传播

### `Embedding`
- **Location**: `sparse.h`
- **Primary Function**: 词嵌入层,将离散的整数索引映射到稠密的连续向量表示,用于处理类别特征或文本 token
- **Key Members**:
  - 无显式成员,权重矩阵存储在 `parameters_["weight"]` 中
- **Core Methods**:
  - `Embedding(num_embeddings, embedding_dim, device)`: 构造函数,创建形状为 `(num_embeddings, embedding_dim)` 的权重矩阵,每个整数索引对应一行嵌入向量,调用 `ResetParameters()` 初始化
  - `Forward(input_tensors)`: 接收整数索引张量(形状任意,如 `(batch_size, seq_len)`),执行查表操作,返回形状为 `(*input_shape, embedding_dim)` 的嵌入张量
  - `ResetParameters()`: 私有方法,初始化权重矩阵,通常使用正态分布 `N(0, 1)`
- **Lifecycle**: 继承自 `CloneableModule<Embedding>`,参数名称常量为 `kParamWeightName="weight"`;典型的参数矩阵大小为 `vocab_size × embedding_dim`

## 3. API Interface

```cpp
// 核心模块基类接口
class Module : public std::enable_shared_from_this<Module> {
    // 类型识别
    const std::string &type() const;

    // 参数管理
    virtual std::vector<std::shared_ptr<Tensor>> Parameters() const;
    bool has_parameter(const std::string &name) const;
    std::shared_ptr<Tensor> *mutable_parameter(const std::string &name);
    const std::shared_ptr<Tensor> &parameter(const std::string &name) const;

    // 缓冲区管理
    virtual std::vector<std::shared_ptr<Tensor>> Buffers() const;

    // 子模块访问
    std::vector<std::shared_ptr<Module>> modules();
    std::shared_ptr<Module> mutable_module(const std::string &name);
    const Module &module(const std::string &name) const;

    // 状态序列化
    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const;

    // 前向传播(纯虚函数,子类必须实现)
    virtual std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors);

    // 训练步骤钩子
    virtual float TrainStep(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors,
        const std::vector<std::shared_ptr<Tensor>> &targets,
        const std::shared_ptr<Module> &loss_fn,
        DataType dtype);

    // 设备和数据类型迁移
    virtual void To(const Device *device);
    virtual void To(DataType dtype);

    // 递归应用函数
    void Apply(std::function<void(Module *)> fn);

    // 数据并行复制
    virtual std::shared_ptr<Module> ReplicateForDataParallel(int device_idx) const;
};

// CRTP 自动克隆接口
template <typename Derived>
class CloneableModule : public Module {
    std::shared_ptr<Module> ReplicateForDataParallel(int device_idx) const override;
};

// 线性层接口
class Linear : public CloneableModule<Linear> {
    Linear(int64_t in_features, int64_t out_features,
           bool bias = true, const Device *device = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    bool bias_;
};

// 容器模块接口
class Sequential : public CloneableModule<Sequential> {
    explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleList : public CloneableModule<ModuleList> {
    explicit ModuleList(std::vector<std::shared_ptr<Module>> &&layers);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    std::vector<std::shared_ptr<Module>>::iterator begin();
    std::vector<std::shared_ptr<Module>>::iterator end();
    std::shared_ptr<Module> &operator[](std::size_t idx);

private:
    std::vector<std::shared_ptr<Module>> module_list_;
};

class ModuleDict : public CloneableModule<ModuleDict> {
    explicit ModuleDict(std::unordered_map<std::string, std::shared_ptr<Module>> modules);
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

// 归一化层接口
class LayerNorm : public CloneableModule<LayerNorm> {
    LayerNorm(const std::vector<int64_t> &normalized_shape,
              float eps = 1e-5f, const Device *device = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    const float eps_;
};

// 嵌入层接口
class Embedding : public CloneableModule<Embedding> {
    Embedding(int num_embeddings, int embedding_dim, const Device *device = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
};
```

## 4. Usage Example

```cpp
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/activations.h"

using namespace infini_train::nn;

// 示例 1: 构建多层感知机(MLP)
void CreateMLP() {
    // 创建线性层: 784 输入特征 -> 256 隐藏单元,启用偏置
    auto fc1 = std::make_shared<Linear>(784, 256, true, device);

    // 创建输出层: 256 -> 10 类别
    auto fc2 = std::make_shared<Linear>(256, 10, true, device);

    // 顺序容器: 依次执行 Linear -> Sigmoid -> Linear
    Sequential mlp({fc1, std::make_shared<Sigmoid>(), fc2});

    // 前向传播
    auto input = std::vector<std::shared_ptr<Tensor>>{CreateInputTensor({32, 784})};
    auto output = mlp.Forward(input);  // 输出形状: (32, 10)

    // 访问参数
    auto weight = fc1->parameter("weight");  // 形状: (256, 784)
    auto bias = fc1->parameter("bias");      // 形状: (256,)

    // 保存模型状态
    auto state_dict = mlp.StateDict();  // {"0.weight": Tensor, "0.bias": Tensor, ...}

    // 迁移到 GPU
    mlp.To(cuda_device);
}

// 示例 2: 构建 Transformer 编码器层
void CreateTransformerLayer() {
    int hidden_size = 512;
    int num_heads = 8;
    int intermediate_size = 2048;

    // 使用 ModuleList 动态管理多层
    auto layers = std::vector<std::shared_ptr<Module>>{};

    for (int i = 0; i < 6; ++i) {
        // 每个编码器层包含 LayerNorm、Linear、Sigmoid 等
        auto norm1 = std::make_shared<LayerNorm>({hidden_size}, 1e-5f, device);
        auto linear1 = std::make_shared<Linear>(hidden_size, intermediate_size, true, device);
        activation = std::make_shared<Sigmoid>();
        auto linear2 = std::make_shared<Linear>(intermediate_size, hidden_size, true, device);

        // 将子模块组合成层
        auto encoder_layer = std::make_shared<Sequential>(
            std::vector<std::shared_ptr<Module>>{norm1, linear1, activation, linear2}
        );

        layers.push_back(encoder_layer);
    }

    // ModuleList 保存所有层
    ModuleList encoder_layers(std::move(layers));

    // 迭代访问每一层
    for (auto &layer : encoder_layers) {
        auto output = layer->Forward(input);
    }
}

// 示例 3: 使用 Embedding 层处理文本
void CreateEmbeddingLayer() {
    int vocab_size = 50000;
    int embedding_dim = 768;
    int max_seq_len = 512;

    // 创建词嵌入矩阵: 50000 个 token,每个映射到 768 维向量
    auto embedding = std::make_shared<Embedding>(vocab_size, embedding_dim, device);

    // 输入: (batch_size=32, seq_len=128) 的整数索引
    auto token_ids = std::vector<std::shared_ptr<Tensor>>{
        CreateTensor({32, 128}, DataType::kInt32)
    };

    // 输出: (32, 128, 768) 的稠密向量
    auto embeddings = embedding->Forward(token_ids);

    // 访问嵌入权重矩阵
    auto weight = embedding->parameter("weight");  // 形状: (50000, 768)
}

// 示例 4: 数据并行复制
void DataParallelExample() {
    // 创建原始模型
    auto model = std::make_shared<Linear>(1000, 100, true, cpu_device);

    // 为每个 GPU 设备创建副本
    std::vector<std::shared_ptr<Module>> model_replicas;
    for (int i = 0; i < 4; ++i) {
        auto replica = model->ReplicateForDataParallel(i);
        replica->To(gpu_devices[i]);
        model_replicas.push_back(replica);
    }

    // 每个副本独立处理不同数据分片
    for (int i = 0; i < 4; ++i) {
        auto shard_output = model_replicas[i]->Forward(data_shards[i]);
    }
}

// 示例 5: 递归操作模块树
void ApplyRecursively() {
    auto model = std::make_shared<Sequential>(/* ... */);

    // 递归设置所有子模块为训练模式
    model->Apply([](Module *m) {
        // 假设 Module 有 SetTrainingMode(bool) 方法
        // m->SetTrainingMode(true);
    });

    // 递归收集所有参数
    auto all_params = model->Parameters();  // 包含所有子模块的参数

    // 递归迁移到指定设备
    model->To(cuda_device);  // 所有参数、缓冲区、子模块都会迁移
}
```

## 5. Implementation Details

**继承体系与多态设计**:
- 使用经典的面向对象继承层次,`Module` 作为抽象基类定义统一接口,所有具体层(Linear/Sigmoid/Embedding等)继承自 `Module`
- 采用 CRTP 模式实现 `CloneableModule<T>`,通过编译期多态避免虚函数开销,为派生类提供自动的 `ReplicateForDataParallel()` 实现
- 前向传播接口使用 `std::vector<std::shared_ptr<Tensor>>` 作为统一输入输出类型,支持多输入多输出模型(如残差连接),代价是需要手动解包

**参数与状态管理**:
- 参数存储在 `std::unordered_map<std::string, std::shared_ptr<Tensor>>` 中,支持字符串键命名访问,便于检查点保存和模块组合
- `Parameters()` 和 `Buffers()` 方法递归遍历模块树,将所有子模块的参数/缓冲区扁平化收集到单个向量中,供优化器统一更新
- `StateDict()` 返回嵌套的字符串-张量映射,键名采用点分隔的层级命名(如 "encoder.layer.0.weight"),兼容 PyTorch 的状态字典格式
- 区分 parameters(可学习梯度参数)和 buffers(非梯度状态如 BatchNorm 的 running_mean),优化器只更新 parameters

**设备抽象与异构计算**:
- 使用 `const Device*` 指针表示计算设备(CUDA/CPU/Kunlun/Metax/Ascend 等),设备对象由外部框架(如 InfiniCore)管理
- `To(const Device*)` 方法递归迁移模块及其所有子模块、参数、缓冲区到目标设备,支持混合设备训练(如模型在 GPU,数据在 CPU)
- 设备指针存储为裸指针而非智能指针,因为设备生命周期通常由运行时全局管理,模块只引用不拥有

**分布式训练支持**:
- 数据并行: 通过 `ReplicateForDataParallel(int device_idx)` 为每个设备创建模型副本,每个副本独立处理不同数据分片,同步梯度
- 流水线并行: 定义特殊常量 `kPPFirstStageName="__pp_first_stage"`, `kPPLastStageName="__pp_last_stage"`, `kPPChunkNamePrefix="__pp_chunk_"` 用于标识流水线阶段
- 声明 `friend std::vector<std::shared_ptr<Module>> parallel::function::Replicate(...)` 允许并行函数访问私有成员进行复杂复制逻辑

**内存管理与所有权**:
- 模块使用 `std::shared_ptr` 管理,支持共享所有权和循环引用(父子模块相互引用)
- 继承 `std::enable_shared_from_this<Module>`,允许成员函数返回指向自身的 `shared_ptr`,避免裸指针悬垂
- 容器构造函数(如 `Sequential(std::vector<Module>&&)`)使用转发引用接管模块所有权,避免拷贝开销
- 参数和缓冲区存储为 `shared_ptr<Tensor>`,允许跨多个模块共享参数(如权重绑定)或创建参数的视图

**初始化策略**:
- 线性层和嵌入层提供 `ResetParameters()` 私有方法,在构造函数中调用以初始化权重
- Linear 层通常使用 Xavier 初始化: `weight ~ U(-sqrt(6/(in+out)), sqrt(6/(in+out)))`,bias 初始化为 0
- Embedding 层通常使用正态分布 `N(0, 1)` 或均匀分布初始化
- LayerNorm 的 weight 初始化为 1,bias 初始化为 0,确保初始阶段接近恒等映射

**容器模块语义差异**:
- `Sequential`: 自动串联执行子模块,`Forward()` 依次调用 `m0->Forward(input) -> m1->Forward(output0) -> ...`,适合固定拓扑
- `ModuleList`: 只提供存储和索引访问,`Forward()` 默认直接返回输入,需要手动遍历调用,适合动态循环层或条件分支
- `ModuleDict`: 支持字符串键查找,`Forward()` 默认直接返回输入,用于专家混合(MoE)或多任务学习路由

**命名空间与依赖隔离**:
- 所有类定义在 `infini_train::nn` 命名空间,避免与其他库冲突
- 前向声明 `class Tensor` 和 `class Device`,减少头文件依赖,加快编译速度
- 依赖外部类型 `DataType` 枚举(定义在 `datatype.h`),支持 float32/float16/bfloat16 等精度
- 所有头文件使用 `#pragma once` 确保单次包含,兼容 MSVC 和现代编译器

**性能优化考虑**:
- 参数查找使用哈希表 `unordered_map`,O(1) 平均查找复杂度,但键名字符串有额外内存开销
- `Apply()` 方法使用函数对象 `std::function<void(Module*)>`,支持 lambda 捕获上下文,但有虚函数间接调用开销
- 克隆操作使用拷贝构造而非序列化/反序列化,避免类型擦除和动态转换开销
- 递归操作(如 `To()`, `Apply()`)可能导致深度嵌套模型的栈溢出风险,但实际模型深度通常 < 100 层,栈空间足够

**错误处理与扩展性**:
- 参数访问 `parameter(name)` 方法未声明 `noexcept`,可能在键不存在时抛出异常或返回空引用(需查看实现)
- 虚函数 `TrainStep()` 提供默认实现返回 0.0f,允许子类可选覆盖,避免强制所有层实现训练钩子
- `type()` 字符串用于运行时类型识别,替代 `dynamic_cast` 进行类型检查,RTTI 可能被禁用以减小二进制体积
- 模块组合通过命名注册支持任意深度嵌套,但未实现循环依赖检测(如 A 的子模块包含 A 会导致无限递归)

**与 PyTorch 的兼容性**:
- API 设计高度模仿 PyTorch 的 `torch.nn.Module`,降低用户迁移成本
- 参数命名约定("weight", "bias")与 PyTorch 一致,便于跨框架模型加载
- 容器类型 `Sequential`, `ModuleList`, `ModuleDict` 提供类似接口,但 `ModuleList` 的迭代器语义略有不同
- `StateDict()` 格式兼容 PyTorch 的分层点分隔命名,支持直接加载 PyTorch 预训练权重(需处理张量格式差异)
