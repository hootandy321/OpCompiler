# Autograd Module Core Implementation Documentation

InfiniTrain 自动微分引擎,基于动态计算图实现可微分张量操作。模块采用函数对象模式(Function Object Pattern),通过前向传播构建计算图,反向传播自动计算梯度,支持分布式训练的梯度通信和多种梯度累积策略。

## 1. Module Structure

- **`function.h`**: 核心抽象基类,定义自动微分函数接口和计算图节点语义
- **`grad_mode.h`**: 梯度计算模式控制,支持全局梯度开关和RAII作用域管理
- **`function_hook.h`**: 梯度后处理钩子机制,实现梯度同步和自定义操作
- **`accumulate.h`**: 梯度累积节点,连接叶节点张量与梯度更新逻辑
- **`activations.h`**: 激活函数微分实现(Sigmoid, Tanh等)
- **`elementwise.h`**: 逐元素运算微分(算术、三角、比较、逻辑运算)
- **`comm.h`**: 分布式通信原语,实现多设备数据分发与梯度聚合
- **`linear.h`**: 线性层微分,权重和偏置梯度计算
- **`matmul.h`**: 矩阵乘法微分,处理链式法则复杂情况
- **`softmax.h`**: Softmax函数微分,结合温度缩放的梯度传播
- **`normalization.h`**: 层归一化微分,维持统计量的梯度修正
- **`loss.h`**: 交叉熵损失微分,集成LogSoftmax数值稳定实现
- **`reduction.h`**: 归约操作微分(Mean/Sum/Max/Min),处理维度变化梯度
- **`transform.h`**: 张量变换微分(转置、三角提取、掩码、重复插值)
- **`misc.h`**: 杂项操作微分(Split/Concat/Stack/Slice/IndexGather)
- **`sparse.h`**: 稀疏操作微分(Embedding查找表梯度)
- **`outer.h`**: 外积运算微分

## 2. Core Classes

### `Function`
- **Location**: `function.h`
- **Primary Function**: 自动微分节点的抽象基类,定义前向传播、反向传播和计算图管理的统一接口。所有可微分操作必须继承此类并实现核心虚函数。
- **Key Members**:
  - `saved_tensors_`: std::vector<std::shared_ptr<Tensor>> - 前向传播保存的张量,用于反向传播梯度计算
  - `next_functions_`: std::vector<std::pair<std::shared_ptr<Function>, int>> - 计算图的后继节点列表,存储子图Function及输入索引
  - `dependencies_number_`: int - 当前节点的输入依赖计数(来自前向传播)
  - `dependencies_reached_`: int - 反向传播到达的依赖计数(用于多输入同步)
  - `grad_outputs_reached_`: int - 梯度输出到达计数(用于多输出同步)
  - `grad_outputs_`: std::vector<std::shared_ptr<Tensor>> - 累积的梯度输出缓冲区
  - `type_`: const std::string - 函数类型标识符(用于调试和序列化)
- **Core Methods**:
  - `Forward(input_tensors)`: 纯虚函数,执行前向计算,输入参数张量,输出结果张量
  - `SetupContext(input_tensors, output_tensors)`: 虚函数(默认空实现),在前向传播后保存反向传播所需中间结果到saved_tensors_
  - `Backward(grad_outputs)`: 纯虚函数,计算相对于输入的梯度,接收输出梯度,返回输入梯度
  - `Apply(input_tensors)`: 组合Forward+SetupContext,自动构建计算图并返回输出张量
  - `BackwardPartial(grad_output, idx)`: 部分梯度处理,将梯度累积到grad_outputs_并检查是否触发完整反向传播
  - `IncreaseDependenciesNumber()`: 增加依赖计数,用于构建多输入计算图的同步逻辑
- **Lifecycle**: 继承自std::enable_shared_from_this<Function>,支持通过shared_from_this()获取自身指针以构建计算图边。通过虚析构函数支持派生类正确清理。类型标识符在构造时初始化,支持RTTI风格的类型检查。

### `GradMode`
- **Location**: `grad_mode.h`
- **Primary Function**: 全局梯度计算开关,控制是否构建计算图和计算梯度。默认启用,可通过作用域守卫临时禁用。
- **Key Members**:
  - `grad_enabled_`: static thread_local bool - 线程局部存储的梯度启用标志,独立线程状态
- **Core Methods**:
  - `IsEnabled()`: 静态方法,返回当前线程的梯度启用状态
  - `SetEnabled(bool)`: 静态方法,设置当前线程的梯度启用状态
- **Lifecycle**: 单例模式(线程局部),静态成员在程序启动时初始化为true(默认启用)。不提供构造/析构,通过静态方法访问。

### `NoGradGuard` / `EnableGradGuard`
- **Location**: `grad_mode.h`
- **Primary Function**: RAII作用域守卫,自动管理梯度模式的临时切换。构造时保存当前状态并设置新状态,析构时恢复原状态。
- **Key Members**:
  - `prev_`: bool - 构造时保存的梯度模式前值
- **Core Methods**:
  - `NoGradGuard()`: 构造函数,保存当前状态并禁用梯度(对应torch.no_grad())
  - `~NoGradGuard()`: 析构函数,恢复构造前状态
  - `EnableGradGuard()`: 构造函数,保存当前状态并强制启用梯度(对应torch.enable_grad())
- **Lifecycle**: 栈对象,作用域内生效,离开作用域自动清理。异常安全(析构函数保证状态恢复)。

### `AccumulateGrad`
- **Location**: `accumulate.h`
- **Primary Function**: 计算图的叶节点,负责将累积的梯度应用到参数张量,触发可选的梯度后处理钩子(如分布式AllReduce)。
- **Key Members**:
  - `tensor_`: std::shared_ptr<Tensor> - 目标参数张量,梯度将被累积到此张量的.grad字段
  - `learning_rate_`: float - 学习率缩放因子,默认1.0(纯梯度累积)
- **Core Methods**:
  - `Forward(input_tensors)`: 前向传播直接返回输入张量(叶节点不修改数据)
  - `Backward(grad_outputs)`: 将梯度乘以学习率并累积到tensor_->grad,触发PostAccumulateGradHook回调
- **Lifecycle**: 在构建计算图时创建,作为叶张量的梯度终点。Backward执行后生命周期结束(由shared_ptr管理)。

### `PostAccumulateGradHook` / `AllReducePostAccumulateHook`
- **Location**: `function_hook.h`
- **Primary Function**: 梯度后处理钩子抽象,支持分布式训练的梯度同步。在AccumulateGrad::Backward后自动调用。
- **Key Members**:
  - `reduce_op_`: ReduceOpType - AllReduce归约操作类型(SUM/AVG/MAX等)
  - `pg_`: const ProcessGroup* - 分布式进程组句柄,控制通信域和设备拓扑
- **Core Methods**:
  - `operator()(tensor)`: 纯虚函数(基类),对tensor执行梯度后处理
  - `operator()(tensor) override`: 实现类,执行ProcessGroup::AllReduce(reduce_op_)同步梯度
- **Lifecycle**: 多态基类,由AccumulateGrad持有shared_ptr。实现类在构造时绑定ProcessGroup和ReduceOpType。

### `Scatter`
- **Location**: `comm.h`
- **Primary Function**: 数据分发,将张量按维度dim分片到多个GPU设备(target_gpus_),支持跨进程通信(ProcessGroup)。
- **Key Members**:
  - `target_gpus_`: std::vector<const Device*> - 目标设备列表,接收分片数据
  - `input_device_`: const Device* - 输入张量所在设备
  - `dim_`: int64_t - 分片维度(通常为batch维度)
  - `pg_`: const ProcessGroup* - 可选的跨进程通信组
- **Core Methods**:
  - `Forward(input_tensors)`: 将输入张量沿dim维度切分为num_chunks,分发到target_gpus
  - `SetupContext()`: 保存输入维度和设备信息
  - `Backward(grad_outputs)`: 收集各设备梯度,沿dim维度拼接回输入形状
- **Lifecycle**: 在数据并行训练中创建,用于分发输入批次到多个GPU。反向传播时自动梯度聚合。

### `Gather`
- **Location**: `comm.h`
- **Primary Function**: 数据收集,将多设备张量沿维度dim聚合到单个目标设备(target_device_)。
- **Key Members**:
  - `target_device_`: const Device* - 聚合目标设备
  - `input_gpus_`: std::vector<const Device*> - 输入张量源设备列表
  - `dim_`: int64_t - 拼接维度
  - `unsqueezed_scalar_`: bool - 标记是否需要额外维度处理(标量广播场景)
  - `pg_`: const ProcessGroup* - 跨进程通信组
- **Core Methods**:
  - `Forward(input_tensors)`: 沿dim拼接多设备输入张量到target_device
  - `Backward(grad_outputs)`: 将梯度切分并分发回各源设备
- **Lifecycle**: 用于收集模型输出或损失到主设备。反向传播时梯度自动回传。

### `Broadcast`
- **Location**: `comm.h`
- **Primary Function**: 数据广播,将张量复制到多个目标设备(target_gpus_),用于同步初始化或广播标量损失。
- **Key Members**:
  - `target_gpus_`: std::vector<const Device*> - 广播目标设备列表
  - `num_inputs_`: int64_t - 输入数量(用于验证)
  - `input_device_`: const Device* - 源设备
  - `pg_`: const ProcessGroup* - 通信组
- **Core Methods**:
  - `Forward(input_tensors)`: 将输入张量复制到所有target_gpus
  - `Backward(grad_outputs)`: 将所有设备梯度归约求和返回源设备
- **Lifecycle**: 用于广播优化器状态或同步梯度。反向传播执行梯度AllReduce(SUM)。

### `ReduceAddCoalesced`
- **Location**: `comm.h`
- **Primary Function**: 梯度融合归约,将多个张量的梯度累加到目标设备(destination_),优化通信带宽。
- **Key Members**:
  - `destination_`: const Device* - 梯度归约目标设备
  - `num_inputs_`: int64_t - 输入张量数量
  - `target_gpus_`: std::vector<const Device*> - 源设备列表
  - `pg_`: const ProcessGroup* - 通信组
- **Core Methods**:
  - `Forward(input_tensors)`: 在target_gpus上执行ReduceAdd到destination
  - `Backward(grad_outputs)`: 将梯度从destination广播回各源设备
- **Lifecycle**: 用于分布式优化器的梯度压缩融合,减少通信次数。

### `Add`
- **Location**: `elementwise.h`
- **Primary Function**: 逐元素加法,支持广播语义。梯度为输入形状的1(链式法则乘法因子为1)。
- **Key Members**:
  - `a_dims_`: std::vector<int64_t> - 第一个操作数的形状(用于反向梯度广播还原)
  - `b_dims_`: std::vector<int64_t> - 第二个操作数的形状
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input[0] + input[1]
  - `SetupContext()`: 保存输入维度用于反向传播处理广播
  - `Backward(grad_outputs)`: 将梯度直接传回输入(可能需要sum reduction处理广播)
- **Complexity**: O(n) 时间复杂度,n为元素数量。梯度传播常数时间。

### `Mul`
- **Location**: `elementwise.h`
- **Primary Function**: 逐元素乘法。梯度需乘以另一个操作数(链式法则: d(a*b)/da = b * grad)。
- **Key Members**:
  - `saved_tensors_`: 通过SetupContext保存前向输入张量
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input[0] * input[1]
  - `SetupContext()`: 保存输入张量到saved_tensors_
  - `Backward(grad_outputs)`: 返回 [grad_outputs * input[1], grad_outputs * input[0]]
- **Complexity**: O(n) 前向和反向。反向传播需2次乘法。

### `Pow`
- **Location**: `elementwise.h`
- **Primary Function**: 幂运算,支持标量指数或标量底数。梯度涉及对数和幂次导数规则。
- **Key Members**:
  - `exponent_`: const float - 幂指数值
  - `scalar_is_base_`: const bool - true表示底数为标量(a^x),false表示指数为标量(x^b)
  - `saved_tensors_`: 保存输入张量用于反向传播
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 pow(input[0], exponent)
  - `SetupContext()`: 保存输入张量
  - `Backward(grad_outputs)`:
    - 标量指数情况: grad * exponent * input^(exponent-1)
    - 标量底数情况: grad * scalar^input * log(scalar)
- **Complexity**: O(n)。反向传播需exp/log运算。

### `Sigmoid`
- **Location**: `activations.h`
- **Primary Function**: Sigmoid激活函数 σ(x)=1/(1+e^-x)。梯度利用前向输出优化计算(σ'(x)=σ(x)*(1-σ(x)))。
- **Key Members**:
  - `saved_tensors_`: 保存前向输出(避免重复计算sigmoid)
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 sigmoid(input)
  - `SetupContext()`: 保存前向输出
  - `Backward(grad_outputs)`: 计算 grad * sigmoid_output * (1 - sigmoid_output)
- **Complexity**: O(n)。反向传播仅需2次算术运算。

### `Tanh`
- **Location**: `elementwise.h`
- **Primary Function**: Tanh激活函数。梯度利用恒等式 tanh'(x)=1-tanh²(x) 优化计算。
- **Key Members**:
  - `saved_tensors_`: 保存前向输出
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 tanh(input)
  - `SetupContext()`: 保存输出
  - `Backward(grad_outputs)`: 计算 grad * (1 - output²)
- **Complexity**: O(n)。反向传播需平方和减法。

### `Matmul`
- **Location**: `matmul.h`
- **Primary Function**: 矩阵乘法,支持2D张量或批处理矩阵乘法。梯度需转置和交换乘法顺序。
- **Key Members**:
  - `out_features_`: int64_t - 输出特征维度(矩阵乘法右侧维度)
  - `saved_tensors_`: 保存输入张量
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input[0] @ input[1]
  - `SetupContext()`: 保存输入张量
  - `Backward(grad_outputs)`:
    - 梯度w.r.t. input[0]: grad @ input[1]^T
    - 梯度w.r.t. input[1]: input[0]^T @ grad
- **Complexity**: O(m*n*p) 前向,m,p,n为三个矩阵维度。反向需两次矩阵乘法。

### `Linear`
- **Location**: `linear.h`
- **Primary Function**: 全连接层 y = xW^T + b。集成权重和偏置的梯度计算,支持可选偏置项。
- **Key Members**:
  - `out_features_`: int64_t - 输出特征维度
  - `bias_`: bool - 是否使用偏置项
  - `saved_tensors_`: 保存输入张量
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input[0] @ input[1]^T + input[2](可选)
  - `SetupContext()`: 保存输入张量
  - `Backward(grad_outputs)`:
    - 输入梯度: grad @ weight
    - 权重梯度: grad^T @ input
    - 偏置梯度: grad沿batch维度sum(如果bias=true)
- **Complexity**: O(in_features * out_features * batch_size)。反向计算量与前向相当。

### `Softmax`
- **Location**: `softmax.h`
- **Primary Function**: Softmax归一化,沿指定维度计算指数归一化。梯度利用Jacobian矩阵性质简化计算。
- **Key Members**:
  - `dim_`: const int64_t - 归一化维度(默认-1,最后一个维度)
  - `saved_tensors_`: 保存前向输出
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 exp(input) / sum(exp(input), dim=dim_, keepdim=True)
  - `SetupContext()`: 保存Softmax输出
  - `Backward(grad_outputs)`: 计算 grad * softmax_output * (1 - softmax_output),沿dim_归约处理交叉项
- **Complexity**: O(n*d) 前向(d为dim维度大小)。反向需额外减法运算。

### `LayerNorm`
- **Location**: `normalization.h`
- **Primary Function**: 层归一化,沿特征维度标准化并应用可学习缩放和偏移。梯度需修正均值和方差的梯度影响。
- **Key Members**:
  - `eps_`: const float - 数值稳定小量(默认1e-5),防止除零
  - `saved_tensors_`: 保存输入、均值、方差、归一化输出
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 (input - mean) / sqrt(var + eps) * gamma + beta
  - `SetupContext()`: 保存输入、均值、方差和标准化输出
  - `Backward(grad_outputs)`:
    - 输入梯度: 结合gamma、beta和统计量的梯度
    - gamma梯度: grad * normalized_output的sum
    - beta梯度: grad的sum
- **Complexity**: O(n)。反向需计算均值和方差的一阶和二阶矩。

### `CrossEntropy`
- **Location**: `loss.h`
- **Primary Function**: 交叉熵损失,集成LogSoftmax和负对数似然。数值稳定实现,避免log(0)。
- **Key Members**:
  - `saved_tensors_`: 保存预测logits和真实标签
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 -log(softmax(logits)[target_class])
  - `SetupContext()`: 保存logits和标签
  - `Backward(grad_outputs)`:
    - logits梯度: softmax_probs - one_hot(target)
    - 标签梯度: 0(离散标签无梯度)
- **Complexity**: O(n * num_classes)。反向传播需重新计算softmax。

### `Mean`
- **Location**: `reduction.h`
- **Primary Function**: 沿指定维度计算均值。反向传播时均匀分配梯度到被归约的维度。
- **Key Members**:
  - `input_dims_`: std::vector<int64_t> - 输入张量形状(用于反向还原)
  - `dim_`: int64_t - 归约维度
  - `keep_dim_`: bool - 是否保持维度(影响反向广播逻辑)
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input.mean(dim=dim_, keepdim=keep_dim_)
  - `SetupContext()`: 保存输入维度
  - `Backward(grad_outputs)`: 将梯度除以归约元素数量并广播回输入形状
- **Complexity**: O(n) 前向。反向需除法和广播操作。

### `Sum`
- **Location**: `reduction.h`
- **Primary Function**: 沿指定维度求和。反向传播直接广播梯度(不缩放)。
- **Key Members**:
  - `input_dims_`: std::vector<int64_t> - 输入张量形状
  - `dim_`: int64_t - 归约维度
  - `keep_dim_`: bool - 维度保持标志
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input.sum(dim=dim_, keepdim=keep_dim_)
  - `SetupContext()`: 保存输入维度
  - `Backward(grad_outputs)`: 直接广播梯度到输入形状
- **Complexity**: O(n) 前向。反向仅需广播操作。

### `Max`
- **Location**: `reduction.h`
- **Primary Function**: 沿指定维度求最大值。反向传播时梯度仅流向最大值元素(类似ReLU路由)。
- **Key Members**:
  - `dim_`: int64_t - 归约维度
  - `keep_dim_`: bool - 维度保持标志
  - `saved_tensors_`: 保存输入张量(用于定位最大值位置)
- **Core Methods**:
  - `Forward(input_tensors)`: 计算 input.max(dim=dim_, keepdim=keep_dim_)
  - `SetupContext()`: 保存输入张量
  - `Backward(grad_outputs)`: 创建零张量,在最大值位置填充梯度
- **Complexity**: O(n) 前向。反向需比较操作定位最大值。

### `Transpose`
- **Location**: `transform.h`
- **Primary Function**: 交换两个维度。反向传播再次交换相同维度恢复原状。
- **Key Members**:
  - `dim0_`: int64_t - 第一个交换维度索引
  - `dim1_`: int64_t - 第二个交换维度索引
- **Core Methods**:
  - `Forward(input_tensors)`: 交换输入张量的dim0_和dim1_维度
  - `Backward(grad_outputs)`: 再次交换dim0_和dim1_维度
- **Complexity**: O(1) 元数据操作(仅修改stride/shape)。

### `Split`
- **Location**: `misc.h`
- **Primary Function**: 沿指定维度将张量切分为多个子张量。反向传播时沿相同维度拼接梯度。
- **Key Members**:
  - `split_size_`: const int64_t - 每个切分块的大小
  - `dim_`: const int - 切分维度
  - `input_dims_`: std::vector<int64_t> - 输入张量形状
- **Core Methods**:
  - `Forward(input_tensors)`: 沿dim_切分输入张量为多个块
  - `SetupContext()`: 保存输入维度
  - `Backward(grad_outputs)`: 沿dim_拼接所有梯度块
- **Complexity**: O(n) 前向和反向(数据复制操作)。

### `Concat`
- **Location**: `misc.h`
- **Primary Function**: 沿指定维度拼接多个张量。反向传播时切分梯度并分发到各输入。
- **Key Members**:
  - `dim_`: const int64_t - 拼接维度
  - `input_dims_list_`: std::vector<std::vector<int64_t>> - 各输入张量的形状列表
- **Core Methods**:
  - `Forward(input_tensors)`: 沿dim_拼接所有输入张量
  - `SetupContext()`: 保存各输入张量形状
  - `Backward(grad_outputs)`: 根据input_dims_list_切分梯度并返回
- **Complexity**: O(n) 前向和反向。

### `Embedding`
- **Location**: `sparse.h`
- **Primary Function**: 嵌入层查找表,根据索引从权重矩阵提取行。反向传播时将梯度累加到对应行(稀疏更新)。
- **Key Members**:
  - `weight_dims_`: std::vector<int64_t> - 权重张量形状[num_embeddings, embedding_dim]
  - `saved_tensors_`: 保存索引张量
- **Core Methods**:
  - `Forward(input_tensors)`: input[0]为索引,input[1]为权重矩阵,执行查找操作
  - `SetupContext()`: 保存索引张量
  - `Backward(grad_outputs)`: 根据索引将梯度scatter_add到权重矩阵
- **Complexity**: O(indices * embedding_dim) 前向。反向需原子累加操作(多线程安全)。

## 3. API Interface

```cpp
// Function类核心接口
class Function {
    virtual std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;
    // 执行前向计算,输入为操作数张量列表,返回结果张量列表

    virtual void SetupContext(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors,
        const std::vector<std::shared_ptr<Tensor>> &output_tensors);
    // 在Forward后调用,保存反向传播所需中间结果到saved_tensors_

    virtual std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) = 0;
    // 计算梯度,输入为输出端的梯度,返回输入端的梯度

    std::vector<std::shared_ptr<Tensor>> Apply(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    // 组合Forward+SetupContext,自动构建计算图并返回输出张量
};

// 梯度模式控制API
class GradMode {
    static bool IsEnabled();
    // 查询当前线程是否启用梯度计算

    static void SetEnabled(bool enabled);
    // 设置当前线程的梯度启用状态
};

// RAII梯度作用域守卫
class NoGradGuard {
    NoGradGuard();
    // 构造时禁用梯度,保存前状态
};

class EnableGradGuard {
    EnableGradGuard();
    // 构造时启用梯度,保存前状态
};

// 梯度后处理钩子接口
class PostAccumulateGradHook {
    virtual void operator()(const std::shared_ptr<Tensor> &tensor) = 0;
    // 纯虚函数,对tensor执行梯度后处理(如AllReduce同步)
};

// 分布式通信函数示例
class Scatter : public Function {
    explicit Scatter(const std::vector<const Device *> &target_gpus, int64_t dim,
                     const infini_train::nn::parallel::ProcessGroup *pg = nullptr);
    // 构造函数: target_gpus为接收分片的设备列表,dim为分片维度,pg为跨进程通信组

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 将输入张量分片到多个GPU设备,返回分片后的张量列表
};

// 逐元素运算函数示例
class Pow : public Function {
    explicit Pow(float exponent, bool scalar_is_base = false);
    // 构造函数: exponent为幂指数,scalar_is_base控制底数/指数谁是标量

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 计算input[0]^exponent或scalar^input[0]
};

// 矩阵乘法函数接口
class Matmul : public Function {
    Matmul();
    // 默认构造函数

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 计算input[0] @ input[1]的矩阵乘法
};

// 线性层接口
class Linear : public Function {
    Linear();
    // 默认构造函数

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 输入张量: [input, weight, bias(可选)],计算 x @ W^T + b
};

// Softmax函数接口
class Softmax : public Function {
    explicit Softmax(int64_t dim = -1);
    // 构造函数: dim为归一化维度,默认-1(最后一个维度)

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 计算exp(input) / sum(exp(input), dim=dim, keepdim=True)
};

// 归约操作接口
class Mean : public Function {
    explicit Mean(int64_t dim, bool keep_dim = false);
    // 构造函数: dim为归约维度,keep_dim控制是否保持维度

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 计算input.mean(dim=dim, keepdim=keep_dim)
};

// 张量变换接口
class Transpose : public Function {
    Transpose(int64_t dim0, int64_t dim1);
    // 构造函数: dim0和dim1为要交换的两个维度索引

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 交换输入张量的dim0和dim1维度
};

// 嵌入层接口
class Embedding : public Function {
    explicit Embedding();
    // 默认构造函数

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    // 输入: [indices, weight],返回weight[indices]的查找结果
};
```

## 4. Usage Example

```cpp
// 示例1: 基础自动微分用法
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"

using namespace infini_train;
using namespace infini_train::autograd;

// 创建可微分张量(假设Tensor类已实现)
auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}); // 输入张量
auto w = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}); // 权重张量

// 构建计算图
Matmul mul_fn;
auto mul_out = mul_fn.Apply({x, w})[0]; // 前向传播: x @ w

Sigmoid sigmoid_fn;
auto sigmoid_out = sigmoid_fn.Apply({mul_out})[0]; // sigmoid(x @ w)

// 反向传播(假设Tensor::backward已实现)
sigmoid_out->backward(); // 自动计算所有前驱节点的梯度

// 访问梯度
auto grad_x = x->grad(); // x的梯度
auto grad_w = w->grad(); // w的梯度

// 示例2: 梯度模式控制
#include "infini_train/include/autograd/grad_mode.h"

// 临时禁用梯度(推理场景)
{
    NoGradGuard no_grad_guard; // RAII,构造时禁用梯度
    auto output = sigmoid_fn.Apply({x})[0]; // 不会构建计算图
    // 推理代码...
} // 析构时自动恢复梯度模式

// 示例3: 分布式训练梯度同步
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/autograd/accumulate.h"

// 创建AllReduce钩子(假设ProcessGroup已初始化)
nn::parallel::ProcessGroup* pg = /* 获取进程组 */;
auto allreduce_hook = std::make_shared<AllReducePostAccumulateHook>(
    nn::parallel::function::ReduceOpType::SUM, pg);

// 创建叶节点并绑定钩子
auto param = std::make_shared<Tensor>(std::vector<int64_t>{128, 256});
AccumulateGrad accum_grad(param, 0.01f /* learning_rate */);
accum_grad.SetHook(allreduce_hook); // 绑定梯度后处理钩子

// 反向传播时自动触发AllReduce
param->backward(); // 梯度会先AllReduce再累积到param->grad

// 示例4: 数据并行训练
#include "infini_train/include/autograd/comm.h"

// 多GPU设置
std::vector<const Device*> gpus = { /* GPU0, GPU1, GPU2, GPU3 */ };

// 分发输入批次到多GPU
Scatter scatter_fn(gpus, /*dim=*/0, /*pg=*/nullptr);
auto scattered_inputs = scatter_fn.Apply({batch_input}); // 返回4个分片张量

// 在各GPU上并行计算
std::vector<std::shared_ptr<Tensor>> outputs;
for (size_t i = 0; i < gpus.size(); ++i) {
    auto out = model_forward(scattered_inputs[i]); // 模型前向传播
    outputs.push_back(out);
}

// 收集梯度到主设备
Gather gather_fn(gpus[0], /*dim=*/0);
auto gathered_grads = gather_fn.Apply(outputs); // 聚合到GPU0

// 示例5: 层归一化使用
#include "infini_train/include/autograd/normalization.h"

LayerNorm layernorm_fn(/*eps=*/1e-5f);
auto hidden_state = std::make_shared<Tensor>(std::vector<int64_t>{32, 128, 512}); // [batch, seq, hidden]
auto gamma = std::make_shared<Tensor>(std::vector<int64_t>{512}); // 可学习缩放
auto beta = std::make_shared<Tensor>(std::vector<int64_t>{512}); // 可学习偏移

auto normed = layernorm_fn.Apply({hidden_state, gamma, beta})[0];
// normalized = (x - mean) / sqrt(var + eps) * gamma + beta

normed->backward(); // 自动计算gamma, beta和输入的梯度

// 示例6: 交叉熵损失
#include "infini_train/include/autograd/loss.h"

CrossEntropy cross_entropy_fn;
auto logits = std::make_shared<Tensor>(std::vector<int64_t>{32, 1000}); // [batch, num_classes]
auto targets = std::make_shared<Tensor>(std::vector<int64_t>{32}); // [batch] 类别索引

auto loss = cross_entropy_fn.Apply({logits, targets})[0]; // 标量损失张量

loss->backward(); // logits梯度自动计算,targets无梯度

// 示例7: 张量拼接与梯度切分
#include "infini_train/include/autograd/misc.h"

Concat concat_fn(/*dim=*/0);
auto part1 = std::make_shared<Tensor>(std::vector<int64_t>{16, 64});
auto part2 = std::make_shared<Tensor>(std::vector<int64_t>{20, 64});
auto part3 = std::make_shared<Tensor>(std::vector<int64_t>{24, 64});

auto concatenated = concat_fn.Apply({part1, part2, part3})[0];
// 结果形状: [60, 64]

concatenated->backward(); // 梯度自动切分为[16,64], [20,64], [24,64]
```

## 5. Implementation Details

- **Memory Management**:
  - 计算图节点使用std::shared_ptr管理生命周期,支持多节点共享同一子图
  - 前向传播中间结果通过saved_tensors_成员保存,避免反向传播重复计算
  - 梯度张量通过Tensor类的.grad成员(std::shared_ptr<Tensor>)持有,支持原地累积
  - 叶节点(AccumulateGrad)负责梯度生命周期管理,非叶节点梯度在反向传播后自动释放

- **Concurrency**:
  - 梯度模式使用thread_local存储,多线程可独立启用/禁用梯度
  - 分布式通信操作通过ProcessGroup抽象,底层支持NCCL/MPI通信后端
  - 梯度累积在AccumulateGrad::Backward中执行,多线程场景需外部同步锁
  - Embedding层的梯度累加使用scatter_add原子操作,保证多线程梯度正确性

- **Performance**:
  - 激活函数(Sigmoid/Tanh)反向传播利用前向输出优化,避免重复exp/log计算
  - Softmax梯度利用Jacobian矩阵稀疏性,简化为逐元素计算避免O(d²)复杂度
  - 分布式操作(Scatter/Gather)通过ProcessGroup支持通信融合(Coalesced),减少延迟
  - 张量变换(Transpose/Slice)使用元数据操作(零拷贝),避免数据复制

- **Error Handling**:
  - 梯度模式错误(禁用状态下调用backward)通过GradMode::IsEnabled()检查
  - 维度不匹配在前向传播时检测,抛出std::invalid_argument异常
  - 设备不一致(CPU/CUDA混合)在通信函数构造时验证
  - 空梯度(leaf node无梯度)通过shared_ptr::nullptr检查

- **Dependencies**:
  - 依赖Tensor类(假设在infini_train命名空间),提供shape/device/data访问接口
  - 依赖Device类表示硬件设备(CUDA/CPU/Kunlun等)
  - 依赖nn::parallel::ProcessGroup提供分布式通信能力
  - 依赖InfiniCore计算算子库执行底层张量运算(add/mul/matmul/softmax等)

- **Design Patterns**:
  - **Strategy Pattern**: Function抽象基类定义算法框架,派生类实现具体微分逻辑
  - **RAII**: NoGradGuard/EnableGradGuard管理资源(梯度模式),保证异常安全
  - **Observer Pattern**: PostAccumulateGradHook钩子机制,解耦梯度计算与同步逻辑
  - **Builder Pattern**: Apply方法组合Forward+SetupContext,简化计算图构建
  - **Template Method**: 模板方法定义反向传播流程(IncreaseDependenciesNumber→BackwardPartial→Backward),派生类实现具体梯度计算
