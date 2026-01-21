# TopKRouter CPU 实现核心文档

TopKRouter CPU 实现是用于专家混合(MoE)模型中的路由算子,实现了基于分组约束的 Top-K 专家选择算法。该算子专门针对 DeepSeek 架构设计,支持从 256 个专家中选择 Top-K 个,并在选择过程中施加分组约束以保证负载均衡。

## 1. 模块结构

- **`topkrouter_cpu.h`**: CPU 实现的描述符声明,使用宏定义扩展顶层接口
- **`topkrouter_cpu.cc`**: CPU 后端完整实现,包含类型转换、Sigmoid 激活、分组 Top-K 选择核心算法

## 2. 核心类与数据结构

### `TopkrouterInfo`
- **位置**: `../info.h`
- **职责**: 封装 TopKRouter 操作的输入张量元信息
- **关键成员**:
  - `xtype`: 输入数据类型(F32/F16/BF16)
  - `shape`: 输入张量形状 [N, width]
  - `x_strides`: 输入张量步长
  - `N`: Batch 维度大小(token 数量)
  - `width`: 专家数量维度(默认 256)
- **约束条件**:
  - 仅支持 2D 张量 `[N, width]`
  - width 必须等于 256
  - width 必须能被 n_group(默认 8)整除
  - 第二维步长必须为 1(连续内存)

### `Descriptor`
- **位置**: `topkrouter_cpu.cc`
- **职责**: TopKRouter CPU 算子的操作描述符,管理算子生命周期和执行
- **继承**: `InfiniopDescriptor`
- **关键成员**:
  - `_opaque`: 不透明指针(保留给扩展,当前为 nullptr)
  - `_info`: TopkrouterInfo 实例,存储张量形状和类型信息
  - `_workspace_size`: 工作空间大小(当前为 0)
- **核心方法**:
  - `create(handle, desc_ptr, x_desc, correction_bias_desc)`: 静态工厂方法,创建描述符并验证输入张量
    - 验证输入张量步长连续性(Stride[1] == 1)
    - 初始化 TopkrouterInfo 元数据
    - 返回 INFINI_STATUS_SUCCESS 或错误码
  - `calculate(...)`: 核心计算方法,执行 TopK 路由算法
    - **时间复杂度**: O(N × width × log(width)) 其中 width=256
    - **空间复杂度**: O(width) 临时存储
    - 支持三种数据类型: F32/F16/BF16
    - 类型分发使用 `if constexpr` 编译期多态

### 辅助数据结构

#### `std::pair<float, size_t>`
- **用途**: 存储(数值, 索引)对,用于排序和索引追踪
- **关键用法**:
  - `value_index_arr`: 存储所有专家的 sigmoid 值和原始索引
  - `value_index_group`: 存储每个组的评分和组索引
  - `value_index_warp`: 组内临时排序数组

## 3. 核心算法实现

### 算法流程总览

TopKRouter CPU 实现采用**四阶段分组约束选择策略**:

```
输入: x [N, 256]  (N 个 token,每个 token 对 256 个专家的原始 logits)
输出: values [N, topk], indices [N, topk]

阶段 1: Sigmoid 激活 + 偏置校正
阶段 2: 分组 Top-2 聚合
阶段 3: 组间选择 Top-K Group
阶段 4: 最终 Top-K 选择 + 归一化
```

### 阶段 1: Sigmoid 激活与偏置校正 (行 51-60)

```cpp
template <typename T>
inline float sigmoid_func(T x) {
    float value;
    if constexpr (std::is_same<T, fp16_t>::value) {
        value = _f16_to_f32(x);  // FP16 → F32 转换
    } else if constexpr (std::is_same<T, bf16_t>::value) {
        value = _bf16_to_f32(x); // BF16 → F32 转换
    } else {
        value = x;               // F32 直接使用
    }
    return 1.0f / (1.0f + std::exp(-value));  // Sigmoid 激活
}
```

**实现细节**:
- 类型转换使用 `if constexpr` 实现编译期分派,零运行时开销
- Sigmoid 函数: `σ(x) = 1 / (1 + e^(-x))`
- 对所有 256 个专家的 logits 逐元素应用 Sigmoid
- 添加 correction_bias 偏置(用于负载均衡校正)

### 阶段 2: 分组 Top-2 聚合 (行 65-84)

**目的**: 将 256 个专家分为 8 个组,每组 32 个专家,计算每组的评分(组内 Top-2 值之和)

```cpp
const size_t group_size = width / n_group;  // 256 / 8 = 32
for (size_t igroup = 0; igroup < n_group; ++igroup) {
    // 提取当前组的 32 个专家
    std::vector<std::pair<float, size_t>> value_index_warp(group_size);
    auto it = value_index_arr.begin() + igroup * group_size;
    for (size_t i = 0; i < group_size; ++i) {
        value_index_warp[i] = {(it++)->first, i};
    }

    // 组内降序排序
    std::sort(value_index_warp.begin(), value_index_warp.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) {
                  return a.first > b.first;
              });

    // Top-2 聚合作为组评分
    value_index_group[igroup] = {
        value_index_warp[0].first + value_index_warp[1].first,  // Top-2 sum
        igroup
    };
}
```

**算法复杂度**:
- 每组排序: O(group_size × log(group_size)) = O(32 × log 32)
- 总复杂度: O(n_group × 32 × log 32) = O(8 × 32 × 5) ≈ O(1280)

**设计意图**:
- **组多样性**: 通过 Top-2 聚合避免单点专家主导,增加组内代表性
- **负载均衡**: 约束最终选中的专家必须分布在不同的组

### 阶段 3: 组间选择 Top-K Group (行 89-97)

**目的**: 从 8 个组中选择评分最高的 Top-K Group(默认 topk_group=4)

```cpp
// 对组评分降序排序
std::sort(value_index_group.begin(), value_index_group.end(),
          [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) {
              return a.first > b.first;
          });

// 生成组掩码
std::vector<bool> group_mask(n_group, false);
for (size_t i = 0; i < topk_group; ++i) {
    size_t index = value_index_group[i].second;
    group_mask[index] = true;  // 标记选中的组
}
```

**复杂度**: O(n_group × log n_group) = O(8 × log 8) ≈ O(24)

**约束效果**:
- 最终选择的专家必须来自这 topk_group 个组
- 未选中组的专家全部置零,无法进入最终选择

### 阶段 4: 组内抑制与最终 Top-K (行 102-141)

#### 步骤 4.1: 组内抑制 (行 102-111)

```cpp
// 将未选中组的所有专家值置 0
for (size_t igroup = 0; igroup < n_group; ++igroup) {
    if (group_mask[igroup]) {
        continue;  // 跳过选中组
    }
    auto it = value_index_arr.begin() + igroup * group_size;
    for (size_t i = 0; i < group_size; ++i) {
        (it++)->first = 0.0f;  // 抑制未选中组
    }
}
```

**复杂度**: O(width) = O(256)

#### 步骤 4.2: 全局 Top-K 选择 (行 116-131)

```cpp
// 对所有 256 个专家(已抑制部分)进行降序排序
std::sort(value_index_arr.begin(), value_index_arr.end(),
          [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) {
              return a.first > b.first;
          });

// 取前 topk 个,计算归一化分母
float exp_sum = 1e-9f;  // 数值稳定的小常数
for (size_t i = 0; i < topk; ++i) {
    size_t index = value_index_arr[i].second;
    float exp_value = sigmoid_func(x_input[index]);  // 重新计算原始 sigmoid

    values_input[i] = exp_value;   // 输出 sigmoid 值(而非偏置后值)
    indices_input[i] = static_cast<int>(index);  // 输出专家索引

    exp_sum += exp_value;
}
```

**关键设计决策**:
- **输出原始 Sigmoid 值**: `values` 输出的是偏置校正前的原始 sigmoid 值,而非经过分组筛选的值
- **避免数值误差**: 重新计算 sigmoid 而非使用已排序的值,保证精度
- **数值稳定性**: exp_sum 初始化为 1e-9 防止除零

#### 步骤 4.3: 归一化 (行 136-140)

```cpp
if (norm_topk_prob) {
    for (size_t i = 0; i < topk; ++i) {
        values_input[i] = routed_scaling_factor * values_input[i] / exp_sum;
    }
}
```

**归一化公式**:
```
output_value = routed_scaling_factor × (sigmoid(x) / Σ sigmoid(x_i))
```

确保 Top-K 个专家的概率和为 `routed_scaling_factor`(通常为 1.0)

## 4. API 接口

### Descriptor::create

```cpp
static infiniStatus_t create(
    infiniopHandle_t handle,                          // InfiniOP 运行时句柄
    Descriptor **desc_ptr,                            // 输出:创建的描述符指针
    infiniopTensorDescriptor_t x_desc,                // 输入张量描述符 [N, 256]
    infiniopTensorDescriptor_t correction_bias_desc   // 偏置张量描述符 [256]
);
```

**返回值**:
- `INFINI_STATUS_SUCCESS`: 创建成功
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型(仅支持 F32/F16/BF16)
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入不是 2D 张量
- `INFINI_STATUS_BAD_TENSOR_STRIDES`: 第二维步长不为 1(非连续内存)

### Descriptor::calculate

```cpp
infiniStatus_t calculate(
    void *workspace,                    // 工作空间(当前未使用,传 nullptr)
    size_t workspace_size,              // 工作空间大小(当前为 0)
    float *values,                      // 输出: Top-K 个专家的权重 [N, topk]
    int *indices,                       // 输出: Top-K 个专家的索引 [N, topk]
    const void *x,                      // 输入: 原始 logits [N, 256]
    const float *correction_bias,       // 输入: 校正偏置 [256]
    const float routed_scaling_factor,  // 缩放因子(通常为 1.0)
    const size_t topk,                  // 选择的专家数量(例如 6)
    void *stream                        // CUDA stream(当前 CPU 实现忽略)
) const;
```

**参数校验**:
- `width` 必须等于 256
- `width % n_group == 0`(256 % 8 == 0)
- 否则返回 `INFINI_STATUS_BAD_PARAM`

**返回值**:
- `INFINI_STATUS_SUCCESS`: 计算成功
- `INFINI_STATUS_BAD_PARAM`: 参数不符合 DeepSeek 配置
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型

## 5. 使用示例

### 基本使用流程

```cpp
#include "infiniop/ops/topkrouter/cpu/topkrouter_cpu.h"

using namespace op::topkrouter::cpu;

// 1. 创建输入张量描述符
infiniopTensorDescriptor_t x_desc;
// 假设 shape: [batch_size=32, n_experts=256]
// dtype: INFINI_DTYPE_F16
createTensorDesc(&x_desc, INFINI_DTYPE_F16, {32, 256});

// 2. 创建偏置张量描述符
infiniopTensorDescriptor_t bias_desc;
createTensorDesc(&bias_desc, INFINI_DTYPE_F32, {256});

// 3. 创建 TopKRouter 描述符
Descriptor* topkrouter_desc;
infiniStatus_t status = Descriptor::create(handle, &topkrouter_desc, x_desc, bias_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 准备输入数据
fp16_t* x = new fp16_t[32 * 256];  // 输入 logits
float* correction_bias = new float[256];  // 偏置

// 5. 准备输出缓冲区
const size_t topk = 6;
float* values = new float[32 * topk];   // 输出权重
int* indices = new int[32 * topk];      // 输出索引

// 6. 执行计算
const float routed_scaling_factor = 1.0f;
status = topkrouter_desc->calculate(
    nullptr,           // workspace (不需要)
    0,                 // workspace_size
    values,            // 输出权重
    indices,           // 输出索引
    x,                 // 输入 logits
    correction_bias,   // 偏置
    routed_scaling_factor,
    topk,              // 选择 6 个专家
    nullptr            // stream (CPU 实现忽略)
);

// 7. 使用结果
for (int n = 0; n < 32; ++n) {
    printf("Token %d 的 Top-%d 专家:\n", n, topk);
    for (size_t k = 0; k < topk; ++k) {
        printf("  专家 %d: 权重 %.4f\n",
               indices[n * topk + k],
               values[n * topk + k]);
    }
}

// 8. 清理资源
delete topkrouter_desc;
delete[] x;
delete[] correction_bias;
delete[] values;
delete[] indices;
```

### DeepSeek V3 MoE 推理示例

```cpp
// DeepSeek V3 配置参数
constexpr size_t n_routed_experts = 256;  // 总专家数
constexpr size_t n_group = 8;              // 分组数
constexpr size_t topk_group = 4;           // 选择的组数
constexpr size_t topk = 6;                 // 每个 token 选择的专家数

// 假设我们有 128 个 token 的 batch
const size_t batch_size = 128;

// 创建描述符
Descriptor* desc;
Descriptor::create(handle, &desc, x_desc, bias_desc);

// 执行路由
desc->calculate(
    nullptr, values, indices,
    logits,           // [128, 256] gating network 输出
    correction_bias,  // [256] 负载均衡偏置
    1.0f,             // routed_scaling_factor
    topk,             // 选择 6 个专家
    nullptr
);

// 结果: 每个 token 被路由到 6 个专家
// 这 6 个专家必须分布在最多 4 个不同的组中
// 约束: (6 个专家) / (4 个组) = 平均每组 1.5 个专家
```

## 6. 实现细节

### 内存管理

**策略**: 纯栈分配 + 临时向量
- **工作空间**: 不需要外部 workspace,`_workspace_size = 0`
- **临时存储**:
  - `value_index_arr`: `std::vector<std::pair<float, size_t>>(width)` ≈ 256 × 8 bytes = 2KB
  - `value_index_group`: `std::vector<std::pair<float, size_t>>(n_group)` ≈ 8 × 8 bytes = 64B
  - `value_index_warp`: `std::vector<std::pair<float, size_t>>(group_size)` ≈ 32 × 8 bytes = 256B
- **每 token 临时内存**: ~2.3KB (栈分配)
- **批量处理**: 循环处理 N 个 token,可复用临时向量

**优势**:
- 无动态内存分配(除 std::vector 内部)
- 缓存友好(小数据集完全在 L1 缓存)
- 无内存碎片

### 并发策略

**当前实现**: 串行处理,无线程并行
```cpp
for (size_t n = 0; n < N; ++n) {
    // 逐个 token 处理
    topkrouter_cpu_one_token<T>(...);
}
```

**并行优化机会**:
1. **Token 级并行**: N 个 token 完全独立,可用 OpenMP 并行
   ```cpp
   #pragma omp parallel for
   for (size_t n = 0; n < N; ++n) {
       topkrouter_cpu_one_token<T>(...);
   }
   ```
2. **SIMD 优化**: Sigmoid 和类型转换可向量化
   ```cpp
   // 使用 AVX2/AVX-512 批量计算 sigmoid
   __m256 x_vec = _mm256_load_ps(x_input);
   __m256 sigmoid_vec = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                      _mm256_add_ps(_mm256_set1_ps(1.0f),
                                                   _mm256_exp_ps(_mm256_sub_ps(_mm256_set1_ps(0.0f), x_vec))));
   ```

### 性能特性

**时间复杂度分析**:
```
每 token 处理时间:
1. 类型转换: O(width) = O(256)
2. Sigmoid + 偏置: O(width) = O(256)
3. 分组排序: O(n_group × group_size × log(group_size)) = O(8 × 32 × 5) = O(1280)
4. 组排序: O(n_group × log(n_group)) = O(8 × 3) = O(24)
5. 组抑制: O(width) = O(256)
6. 全局排序: O(width × log(width)) = O(256 × 8) = O(2048)
7. Top-K 提取: O(topk) = O(6)
8. 归一化: O(topk) = O(6)

总计: O(~4000) 操作/token
```

**吞吐量估算** (单核 CPU):
- 假设 3GHz CPU,每个 token ~4000 操作
- 理论峰值: ~750K tokens/秒
- 实际性能(考虑内存延迟、分支预测): ~100-200K tokens/秒

**优化空间**:
- 使用基数排序或部分选择算法替代完整排序(O(n) → O(n + k log k))
- 预计算 Sigmoid 查找表(LUT,以空间换时间)
- 多线程并行(接近线性加速比)

### 错误处理

**策略**: 静态检查 + 错误码传播
```cpp
// 创建时校验
if (info.x_strides[1] != 1) {
    return INFINI_STATUS_BAD_TENSOR_STRIDES;
}

// 运行时校验
if ((width != n_routed_experts) || (width % n_group != 0)) {
    return INFINI_STATUS_BAD_PARAM;
}

// 类型分发
if (_info.xtype == INFINI_DTYPE_F32) {
    topkrouter_cpu_func<float>(...);
} else if (_info.xtype == INFINI_DTYPE_F16) {
    topkrouter_cpu_func<fp16_t>(...);
} else if (_info.xtype == INFINI_DTYPE_BF16) {
    topkrouter_cpu_func<bf16_t>(...);
} else {
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}
```

**错误类型**:
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量维度错误
- `INFINI_STATUS_BAD_TENSOR_STRIDES`: 内存不连续
- `INFINI_STATUS_BAD_PARAM`: 参数不符合 DeepSeek 配置

**无异常**: 所有错误通过返回码传播,无 C++ 异常抛出

### 数值稳定性

**关键技术**:
1. **Sigmoid 数值稳定**:
   ```cpp
   return 1.0f / (1.0f + std::exp(-value));  // 避免上溢
   ```
   - 对于 `value >> 0`: `exp(-value) ≈ 0`, 结果 ≈ 1.0
   - 对于 `value << 0`: `exp(-value) → ∞`, 结果 ≈ 0.0
   - 未使用额外的数值稳定技巧(如 `max(0, -x)` clipping)

2. **归一化防除零**:
   ```cpp
   float exp_sum = 1e-9f;  // 小常数防止除零
   ```

3. **精确的 Sigmoid 重计算**:
   ```cpp
   float exp_value = sigmoid_func(x_input[index]);  // 使用原始输入重新计算
   ```
   - 避免累积浮点误差
   - 保证输出精度

### 依赖项

**内部依赖**:
- `../topkrouter.h`: 顶层接口定义
- `../../../../utils.h`: 工具类(Result<T>, 宏定义)
- `../../../devices/cpu/common_cpu.h`: CPU 公共定义
- `../../../reduce/cpu/reduce.h`: Reduce 操作(当前未使用)

**外部依赖**:
- `<algorithm>`: `std::sort`
- `<vector>`: `std::vector`

**无第三方库**: 纯标准 C++ 实现

### 设计模式

1. **Template Method Pattern**:
   - `Descriptor` 定义算法骨架
   - `calculate()` 是模板方法,类型分发到模板函数

2. **Strategy Pattern** (编译期):
   - `if constexpr` 实现编译期策略选择
   - 不同数据类型(F32/F16/BF16)有不同的转换策略

3. **Factory Pattern**:
   - `create()` 静态工厂方法创建描述符
   - 封装复杂的初始化和校验逻辑

4. **RAII**:
   - `std::vector` 自动管理临时内存
   - `Descriptor` 析构函数释放资源(当前为空实现)

## 7. 算法约束与假设

### 硬编码参数

当前实现**硬编码** DeepSeek V3 的超参数:
```cpp
const size_t n_routed_experts = 256;  // 总专家数
const size_t n_group = 8;              // 分组数
const size_t topk_group = 4;           // 选择的组数
const bool norm_topk_prob = true;      // 是否归一化
```

**影响**:
- 不支持其他专家数量(如 64, 128)
- 修改需要重新编译
- 灵活性受限但优化潜力大

### 输入约束

1. **张量形状**: 必须为 `[N, 256]`
2. **内存布局**: 第二维步长必须为 1(行优先存储)
3. **数据类型**: 仅支持 F32/F16/BF16
4. **专家数量**: 必须等于 256
5. **分组约束**: `256 % 8 == 0`

### 输出保证

1. **索引唯一性**: Top-K 个专家索引互不相同
2. **索引范围**: 所有索引在 `[0, 255]` 内
3. **权重非负**: 所有输出值 ≥ 0
4. **权重和** (如果归一化): Σ values[i] = routed_scaling_factor
5. **分组约束**: 选中的专家最多来自 topk_group(4) 个不同组

## 8. 应用场景

### DeepSeek V3 MoE 推理

**场景**: 大规模语言模型推理中的专家路由
```cpp
// Gating Network 输出: [batch_size, 256] 每个 token 对每个专家的亲和度
// TopKRouter 输出: [batch_size, 6] 选中的 6 个专家及其权重

// 典型配置
batch_size = 32;  // 32 个 tokens
topk = 6;         // 每个 token 分配到 6 个专家
n_group = 8;      // 256 个专家分为 8 组
topk_group = 4;   // 每个 token 只能使用其中 4 个组的专家

// 结果: 32 × 6 = 192 次专家调用
// 负载均衡约束: 192 次调用分布在 4 个组中,每组平均 48 次
```

**优势**:
- **负载均衡**: 分组约束避免少数专家过载
- **多样性**: 组内 Top-2 聚合增加专家多样性
- **可扩展**: 理论上支持更多专家(需修改代码)

## 9. 已知限制

1. **固定超参数**: 不支持运行时配置专家数量和分组
2. **串行执行**: 未利用多核 CPU 并行
3. **算法复杂度**: 完整排序效率低于部分选择算法
4. **内存效率**: 未使用原地排序,临时向量有拷贝开销
5. **数值精度**: 未使用 GPU 友好的半精度优化(如 Tensor Core)

## 10. 优化建议

### 短期优化

1. **OpenMP 并行化**:
   ```cpp
   #pragma omp parallel for schedule(dynamic)
   for (size_t n = 0; n < N; ++n) {
       topkrouter_cpu_one_token<T>(...);
   }
   ```

2. **使用 `std::partial_sort` 替代完整排序**:
   ```cpp
   // 只需要 Top-K,无需完整排序
   std::partial_sort(value_index_arr.begin(),
                     value_index_arr.begin() + topk,
                     value_index_arr.end(),
                     [](auto &a, auto &b) { return a.first > b.first; });
   ```

3. **预分配临时向量**:
   ```cpp
   // 避免循环内重复分配
   thread_local std::vector<std::pair<float, size_t>> value_index_arr(256);
   ```

### 长期优化

1. **参数化设计**: 将硬编码超参数改为运行时配置
2. **SIMD 向量化**: 使用 AVX-512 批量计算 Sigmoid
3. **GPU Offload**: 使用 CUDA/OpenCL 并行化
4. **算法改进**: 使用基数排序或堆选择降低复杂度

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**作者**: Claude Code Analyst
**适用范围**: InfiniCore/src/infiniop/ops/topkrouter/cpu
