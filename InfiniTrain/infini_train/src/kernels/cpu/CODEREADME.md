# CPU Kernels Core Implementation Documentation

该模块实现了 InfiniTrain 框架的 CPU 后端算子库，提供完整的深度学习训练所需的前向传播和反向传播计算内核。所有算子均采用纯 C++ 实现，支持 float32 精度，部分算子利用 OpenMP 并行化和 Eigen 库优化。

## 1. Module Structure

- **`accumulate_grad.cc`**: 梯度累积与 Adam 优化器更新，支持 OpenMP 并行化
- **`cast.cc`**: 数据类型转换内核，使用模板分发机制
- **`concat.cc`**: 张量拼接操作，基于 memcpy 的高效内存拷贝
- **`cross_entropy.cc`**: 交叉熵损失函数，数值稳定实现（logits 减去最大值）
- **`elementwise.cc`**: 逐元素运算库，包含一元/二元操作及广播机制
- **`embedding.cc`**: 嵌入层查表操作
- **`fill.cc`**: 张量填充操作
- **`gather.cc`**: 索引收集操作（对齐 PyTorch 语义）
- **`layernorm.cc`**: 层归一化，支持 3D 张量 [bs, seq_len, embed_dim]
- **`linear.cc`**: 线性变换与矩阵乘法，使用 Eigen 库加速
- **`no_op.cc`**: 无操作 reshape 占位符
- **`outer.cc`**: 外积运算
- **`reduction.cc`**: 归约操作（mean, sum, max, min）
- **`sigmoid.cc`**: Sigmoid 激活函数
- **`slice.cc`**: 张量切片操作（支持步长）
- **`softmax.cc`**: Softmax 激活函数（数值稳定版本）
- **`split.cc`**: 张量分割操作
- **`stack.cc`**: 张量堆叠操作
- **`transform.cc`**: 张量变换操作（三角矩阵、转置、掩码、重复插值）

## 2. Core Algorithms & Data Structures

### 2.1 梯度更新与优化器 (`accumulate_grad.cc`)

#### `AccumulateGrad`
- **功能**: 简单梯度累积 `tensor += rate * gradient`
- **复杂度**: O(n)，n 为张量元素个数
- **实现**: 串行逐元素更新

#### `AdamAccumulateGrad`
- **功能**: Adam 优化器参数更新
- **关键算法**:
  ```
  m = beta1 * m + (1 - beta1) * grad
  v = beta2 * v + (1 - beta2) * grad^2
  m_hat = m / (1 - beta1^t)
  v_hat = v / (1 - beta2^t)
  param -= lr * m_hat / (sqrt(v_hat) + eps)
  ```
- **并行化**: `#pragma omp parallel for` 并行更新所有参数
- **数值稳定性**: 使用偏差修正项 `bias_correction`

### 2.2 逐元素运算 (`elementwise.cc`)

#### `UnaryForward` / `UnaryBackward`
- **功能**: 一元运算通用模板（neg, reciprocal, sin, cos, tanh, pow, rsqrt, exp, log）
- **实现**: 接受 `std::function<float(float)>` 函数对象，逐元素应用
- **复杂度**: O(n)

#### `BinaryForward` / `BinaryBackward`
- **功能**: 二元运算通用模板（add, sub, mul, div, 比较运算）
- **广播机制**: 支持从低维张量到高维张量的单向广播
- **核心算法**:
  ```cpp
  // 计算 strides 和 padded dimensions
  // 对每个输出元素计算其在输入张量中的索引
  for (int64_t idx = 0; idx < num_elements; ++idx) {
      int64_t b_offset = 0;
      for (int i = 0; i < ndim; ++i) {
          int64_t index = idx / out_strides[i];
          b_offset += (b_padded_dims[i] == 1 ? 0 : index) * b_strides[i];
      }
      out_ptr[idx] = binary_fn(a_ptr[idx], b_ptr[b_offset]);
  }
  ```
- **复杂度**: O(n * ndim)，n 为输出元素个数

### 2.3 矩阵运算 (`linear.cc`)

#### `MatmulForward`
- **功能**: 批量矩阵乘法 `output[*, m, n] = input[*, m, k] * other[*, k, n]`
- **算法**: 三重循环朴素实现
  ```cpp
  for (int64_t b = 0; b < bs; ++b)
      for (int64_t i = 0; i < m; ++i)
          for (int64_t j = 0; j < n; ++j) {
              float acc = 0.0f;
              for (int64_t p = 0; p < k; ++p)
                  acc += input[b*m*k + i*k + p] * other[b*k*n + p*n + j];
              output[b*m*n + i*n + j] = acc;
          }
  ```
- **复杂度**: O(bs * m * n * k)

#### `LinearForward`
- **功能**: 线性层 `output = input * weight^T + bias`（或 weight）
- **优化**: 使用 Eigen 库的矩阵运算
  ```cpp
  output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
  output->EigenMatrix().rowwise() += bias->EigenVector();
  ```
- **复杂度**: Eigen 自动优化（通常 O(bs * in_features * out_features)）

### 2.4 层归一化 (`layernorm.cc`)

#### `LayerNormForward`
- **功能**: 对 3D 张量 [bs, seq_len, embed_dim] 的最后一维归一化
- **算法**:
  ```cpp
  // 对每个 (b, t) 位置
  mean = sum(x) / embed_dim
  variance = sum((x - mean)^2) / embed_dim
  rstd = 1 / sqrt(variance + eps)
  output = (x - mean) * rstd * weight + bias
  ```
- **缓存**: 返回 mean 和 rstd 供反向传播使用
- **复杂度**: O(bs * seq_len * embed_dim)

#### `LayerNormBackward`
- **功能**: 计算输入、权重、偏置的梯度
- **关键公式**:
  ```
  dnorm = grad_output * weight
  dnorm_mean = sum(dnorm) / embed_dim
  dnorm_norm_mean = sum(dnorm * normalized) / embed_dim
  grad_input = (dnorm - dnorm_mean - normalized * dnorm_norm_mean) * rstd
  grad_weight = sum(normalized * grad_output)
  grad_bias = sum(grad_output)
  ```
- **复杂度**: O(bs * seq_len * embed_dim)

### 2.5 交叉熵损失 (`cross_entropy.cc`)

#### `CrossEntropyForward`
- **功能**: 计算交叉熵损失（数值稳定版本）
- **算法**:
  ```cpp
  // 对每个 batch 样本
  max_logit = max(logits)
  sum_exp = sum(exp(logits - max_logit))
  loss -= log(exp(target_logit - max_logit) / sum_exp)
  loss /= batch_size
  ```
- **数值稳定性**: 减去最大值避免 exp 溢出
- **支持类型**: uint8 或 int64 类型的目标索引

#### `CrossEntropyBackward`
- **功能**: 计算 logits 梯度
- **公式**:
  ```
  softmax = exp(logits - max_logit) / sum_exp
  grad_logits = grad_output * (softmax - one_hot(target)) / batch_size
  ```

### 2.6 张量拼接与分割 (`concat.cc`, `split.cc`, `stack.cc`)

#### `ConcatForward`
- **功能**: 沿指定维度拼接多个张量
- **优化**: 使用 `std::memcpy` 批量拷贝连续内存块
  ```cpp
  const int64_t outer_size = product(dims[0:dim])
  const int64_t inner_size = product(dims[dim+1:])
  for (int64_t n = 0; n < outer_size; ++n) {
      for (each input i) {
          memcpy(dst_block, src_ptr + offset, Ki * inner_size * elem_size);
      }
  }
  ```
- **复杂度**: O(total_elements)

#### `SplitForward`
- **功能**: 沿指定维度分割张量为多个块
- **策略**: 按 `split_size` 分割，最后一块可能较小
- **优化**: 同样使用 memcpy 批量拷贝

#### `StackForward`
- **功能**: 沿新维度堆叠多个张量
- **实现**: 类似 concat，但先插入新维度再拷贝数据

### 2.7 归约操作 (`reduction.cc`)

#### `ReduceOpForward`
- **功能**: 通用归约框架（mean, sum, max, min）
- **维度分解**: 将张量分解为 N × H × W，对 H 维度归约
  ```cpp
  int64_t N = product(dims[0:dim])
  int64_t H = dims[dim]
  int64_t W = product(dims[dim+1:])
  for (int64_t n = 0; n < N; ++n)
      for (int64_t w = 0; w < W; ++w)
          output[n*W + w] = reduce_fn(&input[(n*H)*W + w], H);
  ```
- **复杂度**: O(N * H * W)

#### `ReduceOpBackwardMask`
- **功能**: max/min 的反向传播（mask 机制）
- **实现**: 只有最大/最小值位置接收梯度，其他位置为 0

### 2.8 Softmax (`softmax.cc`)

#### `SoftmaxForward`
- **功能**: 沿指定维度计算 softmax
- **数值稳定性**:
  ```cpp
  max_val = max(input_data)
  output = exp(input_data - max_val) / sum(exp(input_data - max_val))
  ```
- **复杂度**: O(outer * axis * inner)

#### `SoftmaxBackward`
- **功能**: Softmax 梯度
- **公式**:
  ```
  dot = sum(y * grad_output)
  grad_input = y * (grad_output - dot)
  ```
- **复杂度**: O(outer * axis * inner)

### 2.9 索引操作 (`gather.cc`, `slice.cc`, `transform.cc`)

#### `IndexGatherForward`
- **功能**: 沿指定维度根据索引收集元素（对齐 PyTorch）
- **索引归一化**: 支持负索引，自动 clamp 到 [0, dim_size-1]
- **实现**: 递归遍历输出张量的所有维度
  ```cpp
  std::function<void(int)> recurse = [&](int d) {
      if (d == num_dims) {
          int64_t gather_j = norm_index[dst_offset];
          src_index[dim] = gather_j;
          out_ptr[dst_offset] = in_ptr[src_offset];
          return;
      }
      for (int64_t i = 0; i < limit; ++i) {
          dst_index[d] = i;
          recurse(d + 1);
      }
  };
  ```

#### `SliceForward`
- **功能**: 支持步长的张量切片
- **参数**: starts, ends, steps（类似 Python 切片）
- **递归实现**: 使用 strides 计算偏移量

#### `TransposeForward`
- **功能**: 交换两个维度
- **Strides 计算**:
  ```cpp
  for (int i = ndim - 2; i >= 0; --i) {
      in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
  }
  ```
- **索引映射**: 输出扁平索引 → 输入多维索引 → 交换 dim0/dim1 → 输出扁平索引

### 2.10 其他算子

#### `EmbeddingForward` (`embedding.cc`)
- **功能**: 查表操作 `output[i] = weight[input[i]]`
- **实现**: 简单双重循环，复制对应的嵌入向量

#### `SigmoidForward` (`sigmoid.cc`)
- **公式**: `y = 1 / (1 + exp(-x))`
- **梯度**: `grad_input = grad_output * y * (1 - y)`

#### `OuterForward` (`outer.cc`)
- **功能**: 外积 `output[i, j] = input[i] * other[j]`
- **优化**: 使用 Eigen 的向量外积

## 3. API Interface

```cpp
// 梯度累积
void AccumulateGrad(
    const std::shared_ptr<Tensor> &gradient,
    float rate,
    const std::shared_ptr<Tensor> &tensor
);

void AdamAccumulateGrad(
    const std::shared_ptr<Tensor> &grad,
    const std::shared_ptr<Tensor> &param,
    const std::shared_ptr<Tensor> &m,
    const std::shared_ptr<Tensor> &v,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    int64_t t
);

// 张量拼接
std::shared_ptr<Tensor> ConcatForward(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    int64_t dim
);

std::vector<std::shared_ptr<Tensor>> ConcatBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<std::vector<int64_t>> &input_dims_list,
    int64_t dim
);

// 矩阵运算
std::shared_ptr<Tensor> MatmulForward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &other
);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MatmulBackward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &other,
    const std::shared_ptr<Tensor> &grad_output
);

std::shared_ptr<Tensor> LinearForward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &weight,
    bool transpose,
    const std::shared_ptr<Tensor> &bias
);

// 层归一化
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormForward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &weight,
    const std::shared_ptr<Tensor> &bias,
    const float eps
);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &weight,
    const std::shared_ptr<Tensor> &bias,
    const std::shared_ptr<Tensor> &mean,
    const std::shared_ptr<Tensor> &rstd,
    const std::shared_ptr<Tensor> &grad_output
);

// Softmax
std::shared_ptr<Tensor> SoftmaxForward(
    const std::shared_ptr<Tensor> &input,
    int64_t dim
);

std::shared_ptr<Tensor> SoftmaxBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::shared_ptr<Tensor> &output,
    int64_t dim
);

// 归约操作
std::shared_ptr<Tensor> MeanForward(
    const std::shared_ptr<Tensor> &input,
    const int64_t dim,
    const bool keep_dim
);

std::shared_ptr<Tensor> SumForward(
    const std::shared_ptr<Tensor> &input,
    const int64_t dim,
    const bool keep_dim
);

std::shared_ptr<Tensor> MaxForward(
    const std::shared_ptr<Tensor> &input,
    const int64_t dim,
    const bool keep_dim
);

// 逐元素运算（示例）
std::shared_ptr<Tensor> AddForward(
    const std::shared_ptr<Tensor> &a,
    const std::shared_ptr<Tensor> &b
);

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<int64_t> &a_dims,
    const std::vector<int64_t> &b_dims
);
```

## 4. Usage Example

```cpp
#include "infini_train/include/tensor.h"
#include "infini_train/include/kernels/cpu.h"

using namespace infini_train;
using namespace infini_train::kernels::cpu;

// 示例 1: 使用 Linear 层进行前向传播
void LinearLayerExample() {
    // 创建输入张量 [batch_size=32, in_features=128]
    auto input_dims = std::vector<int64_t>{32, 128};
    auto input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    input->Fill<float>(1.0f);

    // 创建权重 [out_features=256, in_features=128]
    auto weight_dims = std::vector<int64_t>{256, 128};
    auto weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    weight->Fill<float>(0.01f);

    // 创建偏置 [out_features=256]
    auto bias_dims = std::vector<int64_t>{256};
    auto bias = std::make_shared<Tensor>(bias_dims, DataType::kFLOAT32);
    bias->Fill<float>(0.0f);

    // 前向传播: output = input @ weight^T + bias
    // transpose=true 表示使用 weight^T
    auto output = LinearForward(input, weight, true, bias);
    // output shape: [32, 256]

    // 反向传播（假设有梯度）
    auto grad_output = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32);
    grad_output->Fill<float>(1.0f);

    auto [grad_input, grad_weight, grad_bias] =
        LinearBackward(input, weight, true, 256, grad_output, true);
}

// 示例 2: 层归一化
void LayerNormExample() {
    // 输入: [batch_size=4, seq_len=10, embed_dim=512]
    auto input_dims = std::vector<int64_t>{4, 10, 512};
    auto input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    // 填充数据...

    // 权重和偏置: [embed_dim=512]
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{512}, DataType::kFLOAT32);
    weight->Fill<float>(1.0f);

    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{512}, DataType::kFLOAT32);
    bias->Fill<float>(0.0f);

    // 前向传播
    float eps = 1e-5f;
    auto [output, mean, rstd] = LayerNormForward(input, weight, bias, eps);

    // 反向传播
    auto grad_output = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32);
    grad_output->Fill<float>(1.0f);

    auto [grad_input, grad_weight, grad_bias] =
        LayerNormBackward(input, weight, bias, mean, rstd, grad_output);
}

// 示例 3: 交叉熵损失
void CrossEntropyExample() {
    // 输入 logits: [batch_size=16, num_classes=1000]
    auto logits = std::make_shared<Tensor>(
        std::vector<int64_t>{16, 1000}, DataType::kFLOAT32);
    // 填充随机 logits...

    // 目标标签: [batch_size=16]
    auto targets = std::make_shared<Tensor>(
        std::vector<int64_t>{16}, DataType::kINT64);
    // 填充目标索引...

    // 前向传播计算损失
    auto loss = CrossEntropyForward(logits, targets);
    // loss shape: [] (标量)

    // 反向传播
    auto grad_loss = std::make_shared<Tensor>(
        std::vector<int64_t>{}, DataType::kFLOAT32);
    static_cast<float*>(grad_loss->DataPtr())[0] = 1.0f; // dL/dL = 1

    auto grad_logits = CrossEntropyBackward(logits, targets, grad_loss);
    // grad_logits shape: [16, 1000]
}

// 示例 4: Adam 优化器更新
void AdamOptimizerExample() {
    auto param = std::make_shared<Tensor>(
        std::vector<int64_t>{128, 256}, DataType::kFLOAT32);
    auto grad = std::make_shared<Tensor>(
        std::vector<int64_t>{128, 256}, DataType::kFLOAT32);
    auto m = std::make_shared<Tensor>(
        std::vector<int64_t>{128, 256}, DataType::kFLOAT32);
    auto v = std::make_shared<Tensor>(
        std::vector<int64_t>{128, 256}, DataType::kFLOAT32);

    // 初始化一阶和二阶矩
    m->Fill<float>(0.0f);
    v->Fill<float>(0.0f);

    // Adam 超参数
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    int64_t timestep = 100; // 当前训练步数

    // 执行 Adam 更新
    AdamAccumulateGrad(grad, param, m, v, learning_rate,
                       beta1, beta2, eps, timestep);
}
```

## 5. Implementation Details

### 5.1 并行化策略

#### OpenMP 并行
- **应用算子**: AdamAccumulateGrad 使用 `#pragma omp parallel for`
- **粒度**: 元素级并行，适用于计算密集型操作
- **线程安全**: 每个线程写入独立的内存位置，无竞争

#### SIMD 优化
- **应用算子**: Cast 使用 `#pragma omp parallel for simd`
- **目的**: 利用向量化指令加速逐元素转换

### 5.2 内存管理

#### 内存布局
- **行优先存储**: 所有张量采用 C 风格行优先布局
- **Strides 计算**: 手动计算各维度的步长用于索引映射

#### 高效拷贝
- **memcpy 批量传输**: concat, split, stack 使用 memcpy 处理连续内存块
- **避免逐元素拷贝**: 大块内存拷贝利用 DMA 加速

#### 内存复用
- **NoOp 算子**: 仅改变形状视图，不复制数据
- **Eigen 视图**: Linear/Matmul 使用 Eigen 的矩阵视图避免拷贝

### 5.3 数值稳定性

#### Softmax & Cross Entropy
- **最大值减法**: `exp(x - max(x))` 避免指数溢出
- **对数域计算**: `log(exp(x) / sum(exp)) = x - max - log(sum_exp)`

#### Layer Normalization
- **方差计算**: 使用两轮算法保证精度
  - 第一轮: 计算 mean
  - 第二轮: 计算方差 `sum((x - mean)^2)`
- **epsilon 保护**: `1 / sqrt(variance + eps)` 防止除零

#### Adam 优化器
- **偏差修正**: `m_hat = m / (1 - beta1^t)` 补偿初始化偏差

### 5.4 广播机制

#### 实现细节
- **单向广播**: 仅支持从低维到高维广播（b 广播到 a）
- **对齐检查**: `a->NumElements() % b->NumElements() == 0`
- **维度填充**: 低维张量左侧补 1，例如 `[3] → [1, 1, 3]`
- **步长计算**: 维度为 1 的方向步长为 0，实现索引重复

#### 示例
```cpp
// 广播 [3] 到 [2, 3]
a: [2, 3], b: [3]
b_padded_dims: [1, 3]
b_strides: [0, 1]
// a[0, :] 使用 b[0]
// a[1, :] 使用 b[0]
```

### 5.5 索引操作

#### 负索引处理
- **gather/slice**: 支持类似 Python 的负索引
- **归一化公式**: `idx = idx < 0 ? idx + dim_size : idx`
- **边界检查**: clamp 到 `[0, dim_size-1]`

#### 递归遍历
- **应用场景**: gather, slice 的多维索引计算
- **深度优先**: 递归到最内层后计算偏移量并拷贝数据
- **栈开销**: 对于高维张量可能有递归深度限制

### 5.6 Eigen 库集成

#### 使用场景
- **矩阵乘法**: LinearForward 使用 Eigen 的 GEMM
- **外积**: OuterForward 使用向量外积
- **行/列操作**: `rowwise() += bias` 广播加法

#### 性能优势
- **BLAS 集成**: Eigen 自动调用 BLAS 库（如 OpenBLAS, MKL）
- **表达式模板**: 延迟计算，融合多个操作
- **向量化**: 自动使用 SSE/AVX 指令

### 5.7 错误处理

#### 断言检查
- **形状验证**: 使用 `CHECK` 宏验证张量形状匹配
- **范围检查**: 维度索引、步长合法性验证
- **类型检查**: 数据类型一致性验证

#### 运行时错误
- **溢出保护**: exp, sqrt 的数值溢出未显式处理（依赖 IEEE 754）
- **除零**: LayerNorm 通过 eps 避免，其他算子假设输入合法

### 5.8 性能特征

#### 时间复杂度总结
| 算子 | 复杂度 | 备注 |
|------|--------|------|
| Matmul | O(bs * m * n * k) | 朴素三重循环 |
| Linear | O(bs * in * out) | Eigen 优化 |
| LayerNorm | O(bs * seq * embed) | 两轮遍历 |
| Softmax | O(outer * axis * inner) | 数值稳定版本 |
| Concat | O(total_elements) | memcpy 批量 |
| Elementwise | O(n) | 逐元素 |
| Reduction | O(N * H * W) | 沿 H 维归约 |

#### 空间复杂度
- **原地操作**: 大多数算子创建新输出张量
- **中间缓存**: Softmax/CrossEntropy 复用输出缓冲区
- **梯度张量**: 反向传播创建独立梯度张量

#### 优化建议
- **大批量**: Adam 并行化在大批量下效率高
- **矩阵运算**: 优先使用 Linear 而非 Matmul（Eigen 优化）
- **内存带宽**: concat/slice 的 memcpy 可能成为瓶颈

### 5.9 设计模式

#### 模板分发
- **类型分发**: `DispatchFunc<DataTypeList<...>>` 编译时类型特化
- **避免分支**: 为每种数据类型生成特化代码

#### 函数对象
- **通用接口**: `std::function<float(float)>` 封装运算逻辑
- **代码复用**: UnaryForward/BinaryForward 复用同一框架

#### RAII
- **Tensor 管理**: 使用 `std::shared_ptr` 自动管理生命周期
- **异常安全**: 无显式析构需求

### 5.10 依赖关系

#### 外部库
- **Eigen3**: 矩阵运算（linear.cc, outer.cc）
- **OpenMP**: 并行化（accumulate_grad.cc）
- **glog**: 日志记录（CHECK 宏）
- **STL**: 标准容器和算法

#### 内部模块
- **Tensor 类**: 张量数据结构和操作
- **Dispatcher**: 类型分发和算子注册
- **CPU Common**: 公共工具函数（ComputeStrides, Cast 等）

### 5.11 已知限制

#### 功能限制
- **数据类型**: 大部分算子仅支持 float32
- **维度限制**: LayerNorm 仅支持 3D 张量
- **广播方向**: 仅支持单向广播（低维到高维）
- **步长**: slice 支持步长，但 concat/split 不支持

#### 性能限制
- **朴素实现**: Matmul 未使用分块优化
- **串行执行**: 除 Adam 外大部分算子串行
- **内存拷贝**: 无视图机制，许多操作需拷贝数据

#### 待实现功能
- **共享缓冲区**: Split/Concat 注释提到未来支持 stride 共享内存
- **类型扩展**: Cast 的 TODO 提到支持更多数据类型
- **Batched Outer**: Outer 注释提到未来支持批量外积
