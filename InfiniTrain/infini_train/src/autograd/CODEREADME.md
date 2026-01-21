# Autograd Module Core Implementation Documentation

该模块实现了完整的自动微分系统，支持动态计算图构建、梯度累积和分布式训练场景。核心采用函数式自动微分模式，通过 `Function` 基类统一前向传播和反向传播接口，利用 `Dispatcher` 机制实现硬件后端解耦，并提供了 50+ 种可微分算子的实现。

## 1. Module Structure

- **`accumulate.cc`**: 梯度累加器实现，负责叶节点的梯度累积和优化器更新前的梯度准备
- **`activations.cc`**: 激活函数实现，包括 Sigmoid 及其反向传播
- **`comm.cc`**: 通信原语实现，包括 Scatter、Gather、Broadcast、ReduceAddCoalesced
- **`elementwise.cc`**: 逐元素运算，包含 30+ 种算子（算术、三角、比较、逻辑运算）
- **`function.cc`**: 计算图核心机制，实现前向传播的图构建和反向传播的拓扑执行
- **`function_hook.cc`**: 梯度后处理钩子，支持分布式训练中的 AllReduce 同步
- **`grad_mode.cc`**: 全局梯度模式控制，使用 thread_local 实现线程安全
- **`linear.cc`**: 线性层实现（全连接层）
- **`loss.cc`**: 损失函数实现（CrossEntropy）
- **`matmul.cc`**: 矩阵乘法及反向传播
- **`misc.cc`**: 杂项操作，包括 Split、IndexGather、Slice、Stack、Concat
- **`normalization.cc`**: 归一化层实现（LayerNorm）
- **`outer.cc`**: 外积运算
- **`reduction.cc`**: 归约操作，包括 Mean、Sum、Max、Min
- **`softmax.cc`**: Softmax 激活函数
- **`sparse.cc`**: 稀疏操作实现（Embedding 查找）
- **`transform.cc`**: 张量变换操作，包括 Tril、Triu、Transpose、Mask、RepeatInterleave

## 2. Core Classes

### `Function` (function.cc)
- **Location**: `function.cc`
- **Primary Function**: 计算图节点的抽象基类，管理前向传播时的图构建和反向传播时的梯度流传播
- **Key Members**:
  - `saved_tensors_`: `std::vector<std::shared_ptr<Tensor>>` - 保存前向传播的中间结果，供反向传播使用
  - `next_functions_`: `std::vector<std::pair<std::shared_ptr<Function>, int>>` - 记录当前节点的输入节点及其输出索引
  - `grad_outputs_`: `std::vector<std::shared_ptr<Tensor>>` - 累积来自不同输出路径的梯度
  - `grad_outputs_reached_`: `int` - 已到达的梯度输出数量
  - `dependencies_number_`: `int` - 需要等待的依赖节点数量
  - `dependencies_reached_`: `int` - 已到达的依赖节点数量
- **Core Methods**:
  - `Apply(const std::vector<std::shared_ptr<Tensor>>& input_tensors)`: 前向传播入口，执行 Forward + SetupContext + 图构建
    - 调用虚函数 `Forward()` 计算输出
    - 调用 `SetupContext()` 保存反向传播所需上下文
    - 如果 `GradMode::IsEnabled()`，则为输出张量设置 `grad_fn`，构建计算图
  - `SetupContext(...)`: 虚函数，子类实现以保存反向传播所需的张量
  - `BackwardPartial(const std::shared_ptr<Tensor>& grad_output, int grad_output_idx)`: 部分梯度到达时的处理
    - 使用 accumulator 模式累积多个输出路径的梯度（如果同一节点被多次使用）
    - 当所有梯度输出和依赖都到达时，调用 `Backward()` 并继续反向传播
    - 时间复杂度: O(1) 累积，O(outputs) 反向传播
  - `IncreaseDependenciesNumber()`: 增加依赖计数，用于多输出场景的同步
- **Lifecycle**:
  - 通过 `shared_from_this()` 管理，输出张量持有 `shared_ptr` 到 `grad_fn`
  - 反向传播完成后自动清理 `saved_tensors_` 和 `grad_outputs_`

### `AccumulateGrad` (accumulate.cc)
- **Location**: `accumulate.cc`
- **Primary Function**: 叶节点张量的梯度累加器，连接计算图和优化器
- **Key Members**:
  - `tensor_`: `std::shared_ptr<Tensor>` - 目标叶节点张量
  - `learning_rate_`: `float` - 学习率，用于梯度缩放
- **Core Methods**:
  - `Forward(...)`: 抛出 FATAL 错误，AccumulateGrad 只用于反向传播
  - `Backward(const std::vector<std::shared_ptr<Tensor>>& grad_outputs)`: 梯度累积逻辑
    - 如果张量已有梯度，调用 `AccumulateGrad` kernel 进行累积（支持 `learning_rate_` 缩放）
    - 如果张量没有梯度，创建新梯度张量（使用 `Tensor` 拷贝构造函数切片）
    - 检查 `ConsumeGradOverwriteFlag()`：分布式训练中可能直接覆盖而非累积
    - 调用 `post_accumulate_grad_hook()` 执行自定义后处理（如 AllReduce）
    - 调用 `ResetAccumulator()` 清空累加器状态
- **Implementation Details**:
  - 设备管理：调用 `device->SetDevice()` 确保在正确设备上执行
  - 内存策略：优先复用现有梯度缓冲区，首次分配时拷贝梯度数据
  - 并发安全：通过 `Tensor` 内部锁机制保护梯度读写

### `Elementwise Ops` (elementwise.cc)
包含 30+ 种逐元素运算，统一遵循以下模式：

**Arithmetic Ops**:
- `Neg`, `Reciprocal`, `Sin`, `Cos`, `Tanh`, `Pow`, `Rsqrt`, `Exp`, `Log`
- `Add`, `Sub`, `Mul`, `Div` 及其标量版本（`AddScalar`, `MulScalar` 等）

**Comparison Ops** (不可微):
- `Equals`, `Lt`, `Le`, `Gt`, `Ge`, `Or`, `And` 及其标量版本

**Implementation Pattern**:
```cpp
std::vector<std::shared_ptr<Tensor>> Op::Forward(...) {
    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "OpForward"}, inputs)};
}

void Op::SetupContext(...) {
    saved_tensors_ = {input1, input2, ...}; // 保存反向传播所需张量
}

std::vector<std::shared_ptr<Tensor>> Op::Backward(...) {
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "OpBackward"}, grad_outputs, saved_tensors)};
}
```

**Complexity**: 所有逐元素操作时间复杂度为 O(n)，空间复杂度为 O(n)，n 为张量元素总数

**Memory Strategy**:
- `Mul`, `Div` 等二元操作在 `SetupContext` 中保存两个输入张量
- `Exp`, `Neg` 等一元操作无需保存额外张量，直接计算梯度
- 比较操作的 `Backward()` 抛出 FATAL 错误，因为不应在计算图中出现

### `Sigmoid` (activations.cc)
- **Location**: `activations.cc`
- **Primary Function**: Sigmoid 激活函数及其反向传播
- **Key Members**:
  - `saved_tensors_`: 保存前向传播的输出张量（用于反向传播优化）
- **Core Methods**:
  - `Forward(...)`: 调用 `SigmoidForward` kernel 计算 σ(x) = 1 / (1 + e^(-x))
  - `SetupContext(...)`: 保存输出张量（而非输入），因为梯度公式使用输出值
  - `Backward(...)`: 调用 `SigmoidBackward` kernel，使用输出值计算梯度
    - 梯度公式: ∂L/∂x = ∂L/∂y * y * (1 - y)，其中 y 是 sigmoid 输出
    - 时间复杂度: O(n)

### `Linear` (linear.cc)
- **Location**: `linear.cc`
- **Primary Function**: 全连接层前向和反向传播，支持可选 bias
- **Key Members**:
  - `saved_tensors_`: 保存输入张量和权重张量
  - `bias_`: `bool` - 是否使用 bias
  - `out_features_`: `std::vector<int64_t>` - 输出特征维度
- **Core Methods**:
  - `Forward(input, weight, bias?)`: 计算 `output = input @ weight^T + bias`
    - 调用 `LinearForward` kernel，transpose 参数固定为 true
  - `SetupContext(...)`: 保存 input 和 weight 张量，记录是否使用 bias
  - `Backward(grad_output)`: 计算 `grad_input`, `grad_weight`, `grad_bias`
    - 调用 `LinearBackward` kernel
    - `grad_input` = grad_output @ weight
    - `grad_weight` = grad_output^T @ input
    - `grad_bias` = sum(grad_output, dim=0)（如果存在 bias）
    - 返回元组，根据 `bias_` 决定返回 2 或 3 个梯度
- **Complexity**:
  - 前向: O(m * n * k)，其中 input: (m, k), weight: (n, k), output: (m, n)
  - 反向: O(m * n * k)，三个梯度计算都是矩阵乘法

### `Matmul` (matmul.cc)
- **Location**: `matmul.cc`
- **Primary Function**: 广义矩阵乘法，支持批处理和广播
- **Key Members**:
  - `saved_tensors_`: 保存两个输入张量
  - `out_features_`: `std::vector<int64_t>` - 输出张量维度
- **Core Methods**:
  - `Forward(input1, input2)`: 调用 `MatmulForward` kernel
  - `Backward(grad_output)`: 调用 `MatmulBackward` kernel
    - `grad_input1` = grad_output @ input2^T
    - `grad_input2` = input1^T @ grad_output
    - 返回元组 `(grad_input1, grad_input2)`
- **Complexity**: O(n^3) 对于 n×n 矩阵乘法

### `CrossEntropy` (loss.cc)
- **Location**: `loss.cc`
- **Primary Function**: 交叉熵损失函数，结合了 LogSoftmax 和 NLLLoss
- **Key Members**:
  - `saved_tensors_`: 保存输入张量和目标索引张量
- **Core Methods**:
  - `Forward(input, target)`: 调用 `CrossEntropyForward` kernel
    - input: (N, C) 未归一化的 logits
    - target: (N) 类别索引（0 到 C-1）
    - 输出: 标量损失值
  - `Backward(grad_output)`: 调用 `CrossEntropyBackward` kernel
    - `grad_input` = softmax(input) - one_hot(target)
    - 对 target 返回 nullptr（不需要梯度）
    - 时间复杂度: O(N * C)

### `Softmax` (softmax.cc)
- **Location**: `softmax.cc`
- **Primary Function**: Softmax 激活函数，支持指定维度
- **Key Members**:
  - `dim_`: `int64_t` - Softmax 计算维度
  - `saved_tensors_`: 保存前向传播的输出张量
- **Core Methods**:
  - `Forward(input, dim)`: 调用 `SoftmaxForward` kernel
    - softmax(x_i) = exp(x_i) / Σ exp(x_j)（在指定维度上）
  - `SetupContext(...)`: 保存输出张量（用于优化梯度计算）
  - `Backward(grad_output)`: 调用 `SoftmaxBackward` kernel
    - 使用 Jacobian 矩阵优化公式: ∂L/∂x = y * (∂L/∂y - Σ(∂L/∂y * y))
    - 时间复杂度: O(n * C)，C 为类别数

### `LayerNorm` (normalization.cc)
- **Location**: `normalization.cc`
- **Primary Function**: 层归一化，支持可学习的仿射参数
- **Key Members**:
  - `eps_`: `float` - 数值稳定性的小常数
  - `saved_tensors_`: 保存 input, weight, bias, mean, rstd（5 个张量）
- **Core Methods**:
  - `Forward(input, weight, bias)`: 调用 `LayerNormForward` kernel
    - 返回元组 `(output, mean, rstd)`，保存 mean 和 rstd 用于反向传播
  - `SetupContext(...)`: 将 input, weight, bias 插入到 saved_tensors_ 开头
  - `Backward(grad_output)`: 调用 `LayerNormBackward` kernel
    - 返回元组 `(grad_input, grad_weight, grad_bias)`
    - 梯度计算涉及 mean 和 rstd 的反向传播链式法则
    - 时间复杂度: O(n * d)，n 为 batch size * seq_len，d 为 hidden_dim

### `Embedding` (sparse.cc)
- **Location**: `sparse.cc`
- **Primary Function**: 稀疏嵌入查找表，用于 NLP 中的词嵌入
- **Key Members**:
  - `saved_tensors_`: 保存输入索引张量
  - `weight_dims_`: `std::vector<int64_t>` - 权重表维度
- **Core Methods**:
  - `Forward(indices, weight)`: 调用 `EmbeddingForward` kernel
    - indices: (...,) 整数索引张量
    - weight: (vocab_size, embedding_dim) 嵌入表
    - 输出: (..., embedding_dim) 查找结果
  - `Backward(grad_output)`: 调用 `EmbeddingBackward` kernel
    - 使用稀疏梯度累积，只更新被访问到的嵌入向量
    - 对 input 返回 nullptr（索引不需要梯度）
    - 返回 `(nullptr, grad_weight)`

### `Comm Ops` (comm.cc)
实现分布式训练的通信原语：

**Scatter**:
- `Forward(input, target_gpus, dim)`: 将输入张量沿 dim 维度分散到多个 GPU
- `Backward(grad_outputs)`: 反向传播时执行 Gather，聚合梯度到输入设备

**Gather**:
- `Forward(input_tensors, target_device, dim)`: 从多个设备聚合张量
  - 处理特殊情况：标量张量在维度 0 上 gather 时自动 unsqueeze
- `Backward(grad_outputs)`: 反向传播时执行 Scatter，分发梯度

**Broadcast**:
- `Forward(input, target_gpus)`: 将输入张量广播到多个 GPU（复制）
  - 检查所有输入张量在同一设备类型上
  - 标记为不可微分（TODO）
- `Backward(grad_outputs)`: 执行 ReduceAddCoalesced，梯度求和到源设备

**ReduceAddCoalesced**:
- `Forward(input_tensors, destination)`: 将多个张量分组并归约到目标设备
  - 将输入 reshape 为二维结构 `[device_idx][tensor_idx]`
  - 每组内的 num_inputs 个张量一起归约，减少通信次数
- `Backward(grad_outputs)`: 执行 Broadcast，分发梯度到各源设备

**Performance**:
- 使用 NCCL/通信后端实现集合通信原语
- Coalesced 优化将多次小通信合并为一次大通信

### `Reduction Ops` (reduction.cc)
实现张量归约操作：

**Mean**:
- `Forward(input, dim, keep_dim)`: 计算均值
- `Backward(grad_output)`: 梯度均匀分布到每个元素，除以元素数量

**Sum**:
- `Forward(input, dim, keep_dim)`: 计算和
- `Backward(grad_output)`: 梯度广播到所有位置（使用 shape 恢复）

**Max/Min**:
- `Forward(input, dim, keep_dim)`: 计算最大值/最小值
- `SetupContext(...)`: 保存输入和输出（用于定位最大值位置）
- `Backward(grad_output)`: 只在最大值位置传播梯度，其他位置为 0

**Implementation Pattern**:
```cpp
void ReductionOp::SetupContext(...) {
    input_dims_ = input->Dims(); // 保存原始形状用于梯度恢复
}
```

**Complexity**: O(n) 前向，O(n) 反向（需恢复形状）

### `Transform Ops` (transform.cc)
**Tril/Triu**:
- `Forward(input, diagonal)`: 提取下三角/上三角矩阵
- `Backward(grad_output)`: 梯度只在保留位置非零，其他位置为 0

**Transpose**:
- `Forward(input, dim0, dim1)`: 交换两个维度
- `Backward(grad_output)`: 再次交换相同维度（对称操作）

**Mask**:
- `Forward(input, mask, value)`: 根据 mask 布尔张量将指定位置设为 value
- `Backward(grad_output)`: 只在非 mask 位置传播梯度

**RepeatInterleave**:
- `Forward(input, repeat, dim)`: 沿指定维度重复元素
- `Backward(grad_output)`: 反向重复区域求和

### `Misc Ops` (misc.cc)
**Split**:
- `Forward(input, split_size, dim)`: 沿维度分割张量为多个块
- `Backward(grad_outputs)`: 沿维度拼接所有梯度块

**IndexGather**:
- `Forward(input, index, dim)`: 类似 PyTorch 的 index_select
- `Backward(grad_output)`: 使用 scatter 将梯度写回原始位置

**Slice**:
- `Forward(input, starts, ends, steps)`: 张量切片（类似 Python slicing）
- `SetupContext(...)`: 保存输入张量（获取完整形状）
- `Backward(grad_output)`: 在切片位置填充梯度，其他位置为 0

**Stack**:
- `Forward(input_tensors, dim)`: 沿新维度堆叠张量序列
- `Backward(grad_output)`: 沿维度拆分梯度

**Concat**:
- `Forward(input_tensors, dim)`: 沿已存在维度拼接张量
- `SetupContext(...)`: 保存所有输入张量的维度列表
- `Backward(grad_output)`: 根据记录的维度列表拆分梯度

### `GradMode` (grad_mode.cc)
- **Location**: `grad_mode.cc`
- **Primary Function**: 全局梯度计算开关，使用 RAII 模式管理
- **Key Members**:
  - `grad_enabled_`: `thread_local bool` - 每线程独立状态，默认 true
- **Core Methods**:
  - `IsEnabled()`: 静态方法，返回当前线程的梯度模式
  - `NoGradGuard`: RAII 作用域守卫，构造时禁用梯度，析构时恢复
- **Thread Safety**: 使用 `thread_local` 保证多线程环境下的独立性

### `AllReducePostAccumulateHook` (function_hook.cc)
- **Location**: `function_hook.cc`
- **Primary Function**: 梯度累积后的自定义钩子，支持分布式 AllReduce
- **Key Members**:
  - `reduce_op_`: `ReduceOpType` - 归约操作类型（SUM, AVG, MAX, MIN）
  - `pg_`: `const ProcessGroup*` - 进程组句柄
- **Core Methods**:
  - `operator()(tensor)`: 函数调用操作符，执行 AllReduce
    - 调用 `parallel::function::AllReduce(tensor, reduce_op, pg)`
    - 用于分布式数据并行训练中的梯度同步
- **Usage Pattern**:
  ```cpp
  tensor->set_post_accumulate_grad_hook(std::make_shared<AllReducePostAccumulateHook>(ReduceOpType::kSum));
  ```

## 3. API Interface

```cpp
// 计算图构建 API
std::vector<std::shared_ptr<Tensor>> Function::Apply(const std::vector<std::shared_ptr<Tensor>>& input_tensors);
// 前向传播入口，构建计算图。如果 GradMode::IsEnabled()，为输出张量设置 grad_fn

// 反向传播触发 API（通常在 Tensor 中实现，但由 Function 调用）
void Function::BackwardPartial(const std::shared_ptr<Tensor>& grad_output, int grad_output_idx);
// 部分梯度到达时的处理，支持多输出场景的梯度累积

// 梯度模式控制 API
class GradMode {
public:
    static bool IsEnabled();  // 查询当前梯度模式
    static void set_enabled(bool enabled);  // 设置梯度模式

    class NoGradGuard {  // RAII 作用域守卫
        NoGradGuard();
        ~NoGradGuard();
    };
};

// Function 子类接口（由各算子实现）
class Function {
protected:
    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>>& input_tensors) = 0;
    virtual void SetupContext(const std::vector<std::shared_ptr<Tensor>>& inputs,
                             const std::vector<std::shared_ptr<Tensor>>& outputs) {}
    virtual std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>>& grad_outputs) = 0;
};

// 通信原语 API
class Scatter : public Function {
public:
    Scatter(const std::vector<const Device*>& target_gpus, int64_t dim, const ProcessGroup* pg = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(...);  // 返回多个张量，每个对应一个 GPU
};

class Gather : public Function {
public:
    Gather(const Device* target_device, int64_t dim, const ProcessGroup* pg = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(...);  // 从多设备聚合到单个张量
};

// 钩子 API
class AllReducePostAccumulateHook {
public:
    AllReducePostAccumulateHook(ReduceOpType reduce_op, const ProcessGroup* pg = nullptr);
    void operator()(const std::shared_ptr<Tensor>& tensor);  // 在梯度累积后调用
};
```

## 4. Usage Example

```cpp
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/loss.h"
#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
using namespace infini_train::autograd;

// 示例：构建一个简单的全连接神经网络前向传播
void SimpleNeuralNetwork() {
    // 1. 创建输入张量和参数（标记 requires_grad = true）
    auto input = std::make_shared<Tensor>(TensorShape{32, 784}, DeviceType::kCUDA);
    input->set_requires_grad(false);  // 输入不需要梯度

    auto weight1 = std::make_shared<Tensor>(TensorShape{128, 784}, DeviceType::kCUDA);
    weight1->set_requires_grad(true);  // 参数需要梯度
    weight1->set_grad_accumulator(std::make_shared<AccumulateGrad>(weight1, 1.0f));

    auto bias1 = std::make_shared<Tensor>(TensorShape{128}, DeviceType::kCUDA);
    bias1->set_requires_grad(true);
    bias1->set_grad_accumulator(std::make_shared<AccumulateGrad>(bias1, 1.0f));

    // 2. 构建计算图（前向传播）
    // Linear 层: y = xW^T + b
    auto linear_fn = std::make_shared<Linear>();
    auto hidden_outputs = linear_fn->Apply({input, weight1, bias1});
    auto hidden = hidden_outputs[0];  // shape: (32, 128)

    // Sigmoid 激活
    auto sigmoid_fn = std::make_shared<Sigmoid>();
    auto activated = sigmoid_fn->Apply({hidden})[0];

    // 第二个线性层
    auto weight2 = std::make_shared<Tensor>(TensorShape{10, 128}, DeviceType::kCUDA);
    weight2->set_requires_grad(true);
    weight2->set_grad_accumulator(std::make_shared<AccumulateGrad>(weight2, 1.0f));

    auto output = std::make_shared<Linear>()->Apply({activated, weight2})[0];

    // 3. 计算损失（CrossEntropy）
    auto target = std::make_shared<Tensor>(TensorShape{32}, DeviceType::kCUDA);  // 类别索引
    target->set_requires_grad(false);

    auto loss_fn = std::make_shared<CrossEntropy>();
    auto loss = loss_fn->Apply({output, target})[0];  // 标量张量

    // 4. 反向传播（自动计算所有梯度）
    // 从损失张量触发反向传播
    auto grad_loss = std::make_shared<Tensor>(TensorShape{1}, DeviceType::kCUDA);
    grad_loss->Fill(1.0f);  // dL/dL = 1.0

    loss->grad_fn()->BackwardPartial(grad_loss, loss->output_idx());

    // 5. 访问计算出的梯度
    auto grad_weight1 = weight1->grad();  // 梯度已自动累积
    auto grad_bias1 = bias1->grad();
    auto grad_weight2 = weight2->grad();

    // 6. 使用梯度更新参数（在优化器中）
    // optimizer->Step();  // 会清空梯度缓冲区
}

// 示例：禁用梯度计算（推理模式）
void InferenceMode() {
    autograd::GradMode::set_enabled(false);  // 全局禁用

    auto input = std::make_shared<Tensor>(TensorShape{1, 784}, DeviceType::kCUDA);
    auto weight = std::make_shared<Tensor>(TensorShape{10, 784}, DeviceType::kCUDA);
    weight->set_requires_grad(true);  // 即使设置了 requires_grad，也不会构建图

    auto output = std::make_shared<Linear>()->Apply({input, weight})[0];
    // output->grad_fn() == nullptr，因为 GradMode 被禁用

    autograd::GradMode::set_enabled(true);  // 恢复梯度模式
}

// 示例：使用 NoGradGuard 临时禁用梯度
void TemporarilyDisableGrad() {
    auto input = std::make_shared<Tensor>(TensorShape{1, 784}, DeviceType::kCUDA);
    auto weight = std::make_shared<Tensor>(TensorShape{10, 784}, DeviceType::kCUDA);

    {
        autograd::GradMode::NoGradGuard no_grad_guard;  // 进入作用域时禁用
        auto output = std::make_shared<Linear>()->Apply({input, weight})[0];
        // 此时不构建计算图
    }  // 离开作用域时自动恢复梯度模式

    // 现在梯度模式已恢复
}

// 示例：分布式训练中的梯度同步
void DistributedTrainingWithHook() {
    auto weight = std::make_shared<Tensor>(TensorShape{128, 784}, DeviceType::kCUDA);
    weight->set_requires_grad(true);

    auto accum_grad = std::make_shared<AccumulateGrad>(weight, 1.0f);
    weight->set_grad_accumulator(accum_grad);

    // 设置梯度累积后的 AllReduce 钩子
    auto allreduce_hook = std::make_shared<AllReducePostAccumulateHook>(
        infini_train::nn::parallel::function::ReduceOpType::kSum,
        infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()
    );
    weight->set_post_accumulate_grad_hook(allreduce_hook);

    // ... 前向传播和反向传播 ...

    // 当梯度累积到 weight 后，会自动调用 AllReduce 钩子同步梯度
    // accum_grad->Backward(...) 内部会调用 hook
}
```

## 5. Implementation Details

### 计算图构建机制 (Function::Apply)
- **拓扑结构**: 采用动态图模式，前向传播时即时构建计算图
- **节点管理**: 每个 `Function` 实例代表计算图中的一个节点
- **边关系**: 通过 `next_functions_` 记录依赖边，存储 `(next_function, output_idx)` 对
- **叶节点识别**: 使用 `is_leaf()` 标记，叶节点的 `grad_fn` 为 nullptr，但有 `grad_accumulator`
- **梯度流**: 反向传播时从输出节点沿 `next_functions_` 逆向遍历

### 梯度累积策略 (AccumulateGrad)
- **累积模式**: 支持多次反向传播累积梯度到同一张量（适用于小批量训练）
- **内存优化**: 首次分配梯度缓冲区后复用，避免频繁分配
- **分布式优化**: `ConsumeGradOverwriteFlag` 机制支持梯度覆盖而非累积（减少通信）
- **钩子机制**: `post_accumulate_grad_hook` 允许在累积后执行自定义逻辑（如 AllReduce）

### 反向传播执行 (BackwardPartial)
- **多输出同步**: 使用计数器 `grad_outputs_reached_` 等待所有输出路径的梯度到达
- **依赖计数**: `dependencies_number_` 和 `dependencies_reached_` 管理多输入同步
- **梯度累加**: 同一输出多次到达时使用 `AccumulateGrad` kernel 求和
- **拓扑排序**: 递归调用 `next_function->BackwardPartial()` 实现深度优先反向遍历
- **内存释放**: 反向传播完成后立即清理 `saved_tensors_` 和 `grad_outputs_`，节省内存

### Dispatcher 模式
- **硬件解耦**: 所有算子通过 `Dispatcher::Instance().Call<ReturnType>({device_type, op_name}, args)` 调用
- **后端注册**: 不同硬件（CUDA, CPU, KUNLUN, METAX 等）注册各自的 kernel 实现
- **统一接口**: `Function` 层与硬件后端完全解耦，添加新硬件只需注册新 kernel
- **类型安全**: 使用模板参数 `ReturnType` 支持多返回值（张量、元组、向量）

### 内存管理
- **张量生命周期**: 使用 `std::shared_ptr` 管理，避免手动释放
- **循环引用**: `Function` 通过 `shared_from_this()` 避免循环引用，输出张量持有 `grad_fn` 的 `weak_ptr` 或原始指针
- **上下文保存**: `saved_tensors_` 只在反向传播时需要，完成后立即清空
- **梯度缓冲区**: `AccumulateGrad` 复用梯度缓冲区，减少内存分配开销

### 并发安全
- **Thread-Local GradMode**: `GradMode::grad_enabled_` 使用 `thread_local`，每线程独立
- **设备上下文**: 每次操作前调用 `device->SetDevice()` 确保在正确设备上执行
- **梯度锁**: `Tensor` 内部使用锁机制保护 `grad_` 的并发读写
- **分布式同步**: 通过 `ProcessGroup` 和 `AllReducePostAccumulateHook` 实现多机多卡同步

### 性能优化
- **Kernel Fusion**: Dispatcher 支持后端实现 fused kernel（如 LayerNormForward 同时计算 output, mean, rstd）
- **Coalesced Communication**: `ReduceAddCoalesced` 将多个小通信合并为一次大通信
- **梯度覆盖**: 分布式训练中使用 `ConsumeGradOverwriteFlag` 避免梯度累积后的额外 AllReduce
- **延迟计算**: 只在反向传播时保存必要张量，前向传播不保存中间结果（除非 `SetupContext` 显式保存）

### 错误处理
- **CHECK 宏**: 使用 `CHECK_EQ`, `CHECK_GE` 等宏进行前置条件断言
- **FATAL 错误**: 不应调用的方法（如 `AccumulateGrad::Forward`）抛出 `LOG(FATAL)`
- **空指针检查**: 梯度张量可能为 nullptr（如 `CrossEntropy` 对 target 的梯度），在反向传播中需检查

### 设计模式
- **Strategy Pattern**: Dispatcher 根据设备类型选择不同的 kernel 实现
- **RAII**: `NoGradGuard` 使用构造/析构函数自动管理梯度模式
- **Observer**: `Function` 的 `next_functions_` 实现观察者模式，梯度流动触发下游节点
- **Template Method**: `Apply` 定义算法骨架，子类实现 `Forward`, `Backward` 等虚函数
- **Command Pattern**: 钩子机制使用函数对象封装自定义操作

### 复杂度保证
- **逐元素操作**: O(n) 时间，O(n) 空间
- **矩阵乘法**: O(m * n * k) 时间，O(m * n + n * k) 空间（保存输入用于反向）
- **归约操作**: O(n) 时间，O(n) 空间（需保存输入形状）
- **Softmax/LogSoftmax**: O(n * C) 时间，O(n) 空间
- **LayerNorm**: O(n * d) 时间，O(n * 3) 空间（保存 input, weight, bias, mean, rstd）

### 依赖关系
- **外部依赖**: `glog`（日志），`infini_train::Tensor`（张量抽象），`infini_train::Device`（设备抽象）
- **模块依赖**:
  - `autograd/comm.h` 依赖 `nn/parallel/process_group.h`（分布式通信）
  - `autograd/function_hook.h` 依赖 `nn/parallel/parallel_functional.h`（AllReduce 实现）
  - 所有算子依赖 `dispatcher.h`（kernel 调度）
- **硬件后端**: 通过 Dispatcher 间接依赖 CUDA, CPU, KUNLUN 等后端实现

### 扩展性
- **添加新算子**: 继承 `Function`，实现 `Forward`, `SetupContext`, `Backward` 三个虚函数
- **添加新硬件**: 在 Dispatcher 中注册新的 kernel 实现，无需修改 `autograd` 层代码
- **自定义梯度**: 重写算子的 `Backward` 方法实现特定梯度计算逻辑
- **钩子扩展**: 实现自定义 `FunctionHook` 子类，支持梯度后处理（如量化、裁剪）
