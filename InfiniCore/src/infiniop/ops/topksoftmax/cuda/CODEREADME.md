# TopK-Softmax CUDA Kernel 核心实现文档

该模块实现了 CUDA 加速的 TopK-Softmax 融合操作内核，在单个 CUDA kernel 中完成 Softmax 归一化、TopK 选择和可选的二次归一化，专为大型语言模型推理中的采样操作优化。

## 1. 模块结构

- **`kernel.cuh`**: CUDA kernel 头文件，包含 `softmax_topk_row_kernel` 模板函数及配套设备函数，实现 TopK-Softmax 融合计算

## 2. 核心函数

### `exp_func<T>`
- **位置**: `kernel.cuh:10-20`
- **主要功能**: 类型安全的指数函数设备函数，将不同数据类型统一转换为 `float` 后计算 `exp(x)`
- **模板参数**: `T` - 支持的类型包括 `float`、`cuda_bfloat16`、`half`
- **实现细节**:
  - 使用 `if constexpr` 编译期类型分发
  - `float` 类型直接使用原值
  - `cuda_bfloat16` 通过 `__bfloat162float()` 内置函数转换
  - `half` 通过 `__half2float()` 内置函数转换
  - 最终调用 `__expf()` CUDA 内置指数函数
- **时间复杂度**: O(1)

### `softmax_topk_row_kernel<T, BLOCK_SIZE>`
- **位置**: `kernel.cuh:22-144`
- **主要功能**: 融合 Softmax + TopK 选择 + 可选归一化的行级操作，对输入矩阵的每一行独立计算 Softmax 分布，提取 TopK 个最大值及其索引
- **模板参数**:
  - `T`: 输入数据类型（支持 float/half/bfloat16）
  - `BLOCK_SIZE`: CUDA block 线程数，默认 128
- **核心成员**:
  - `values_topk`: 输出 TopK 值数组，形状 [N, topk]
  - `indices_topk`: 输出 TopK 索引数组，形状 [N, topk]
  - `input`: 输入数据，形状 [N, width]
  - `N`: 处理的行数（block 数量）
  - `width`: 每行的维度（必须 <= BLOCK_SIZE）
  - `topk`: 提取的 TopK 数量（必须 <= 32）
  - `norm`: 是否进行二次归一化的布尔标志
- **共享内存**:
  - `shared_max`: 存储每行最大值（类型 T）
  - `shared_sum`: 存储指数和或 TopK 和（类型 float）
  - `temp_storage_max/sum/sort`: CUB 原语临时存储
- **执行流程** (7 个阶段):

  1. **计算最大值** (第49-67行):
     - 每个 thread 先计算本地最大值
     - 使用 `cub::BlockReduce` 进行跨 block 归约求全局最大值
     - 兼容 CUDA 12.9+ 的 `cuda::maximum()` 和旧版的 `cub::Max()`

  2. **计算指数和** (第69-85行):
     - 计算 `exp(input[i] - max)` 以提高数值稳定性
     - 使用 `BlockReduce::Sum()` 归约求和

  3. **Softmax 归一化** (第87-90行):
     - 将每个指数值除以总和，得到概率分布

  4. **TopK 排序** (第92-106行):
     - 每个 thread 准备单个键值对 `(value, index)`
     - 使用 `cub::BlockRadixSort` 降序排序，复杂度 O(log n)
     - 排序后 block 内按 softmax 概率降序排列

  5. **Warp 级 TopK 求和** (第108-127行):
     - 仅第一个 warp (warp_id == 0) 参与
     - 提取前 `topk` 个元素
     - 使用 `cub::WarpReduce` 计算这些值的和
     - 添加 `1e-9f` 小常数避免除零

  6. **二次归一化** (第129-134行):
     - 如果 `norm == true`，将 TopK 值除以它们的和
     - 使 TopK 结果重新归一化为概率分布

  7. **写入结果** (第136-143行):
     - 前 `topk` 个 thread 将值和索引写入全局内存

## 3. API 接口

```cpp
template <typename T, int BLOCK_SIZE = 128>
__global__ void softmax_topk_row_kernel(
    float *values_topk,        // 输出: TopK 概率值 [N, topk]
    int *indices_topk,         // 输出: TopK 索引 [N, topk]
    const T *input,            // 输入: logits [N, width]
    const size_t N,            // 行数
    const size_t width,        // 每行维度 (width <= BLOCK_SIZE)
    const size_t topk,         // TopK 数量 (topk <= 32)
    bool norm                  // 是否二次归一化
);
// 功能: 融合 Softmax + TopK + 可选归一化
// 约束: width <= BLOCK_SIZE, topk <= 32
```

## 4. 使用示例

```cpp
// 示例: LLM 推理中的 top-k 采样
// 假设 vocab_size = 32000, batch_size = 1, topk = 50

const int N = 1;              // batch size
const int width = 32000;      // 词汇表大小
const int topk = 50;          // top-k 采样参数
const int BLOCK_SIZE = 128;   // CUDA block 大小

// 分配内存
float* d_values;
int* d_indices;
half* d_logits;

cudaMalloc(&d_values, N * topk * sizeof(float));
cudaMalloc(&d_indices, N * topk * sizeof(int));
cudaMalloc(&d_logits, N * width * sizeof(half));

// 启动 kernel (每个 block 处理一行)
dim3 grid(N);
dim3 block(BLOCK_SIZE);
softmax_topk_row_kernel<half, BLOCK_SIZE>
    <<<grid, block>>>(
        d_values,      // 输出 TopK 概率
        d_indices,     // 输出 TopK token 索引
        d_logits,      // 输入 logits
        N,             // 1 行
        width,         // 32000 维
        topk,          // 取前 50
        true           // 归一化使这些概率和为 1
    );

// 在 CPU 端基于 TopK 概率进行采样
cudaMemcpyAsync(h_values, d_values, ...);
cudaMemcpyAsync(h_indices, d_indices, ...);
// 然后使用 multinomial 采样从 topk 分布中选择 token
```

## 5. 实现细节

**融合优化策略**:
- **单 Kernel 执行**: 将 Softmax、排序、TopK 提取三个阶段融合为一个 kernel，减少全局内存访问
- **数值稳定性**: 通过 `exp(x - max)` 避免 float 溢出，这是 Softmax 实现的标准技巧
- **CUB 原语集成**: 使用 NVIDIA CUB 库的高性能 block/warp 级原语
  - `BlockReduce`: O(log n) 归约，比原子操作快数倍
  - `BlockRadixSort`: 高效基数排序，比比较排序快
  - `WarpReduce`: warp 内快速归约，延迟极低

**内存访问模式**:
- **合并访问**: 所有线程按顺序访问 `input[tid]`，满足合并读取条件
- **共享内存复用**: `shared_max` 和 `shared_sum` 在不同阶段复用，减少共享内存占用
- **写回限制**: 仅前 `topk` 个线程写入全局内存，减少写带宽

**并行策略**:
- **Block 维度**: 每个 block 处理矩阵的一行
- **Warp 优化**: TopK 提取仅使用第一个 warp，减少分歧
- **约束条件**:
  - `width <= BLOCK_SIZE`: 保证每个元素有对应线程
  - `topk <= 32`: 保证单个 warp 可处理

**类型支持**:
- 编译期模板特化处理 `float`、`half` (FP16)、`cuda_bfloat16` (BF16)
- 通过 `if constexpr` 零运行时开销的类型转换
- 内部计算统一使用 `float` 精度

**性能特性**:
- 时间复杂度: O(width + log(BLOCK_SIZE)) 每行
- 空间复杂度: O(BLOCK_SIZE) 共享内存
- 带宽需求: 读取 N*width 输入，写入 N*topk 输出（当 width >> topk 时大幅减少）

**二次归一化 (norm 参数)**:
- 当 `norm=true` 时，TopK 概率和被强制为 1.0
- 用于需要从 TopK 子集重新采样的场景（如 nucleus sampling 的变体）
- 添加 `1e-9f` 常数防止除零，这在极端情况下（所有概率接近 0）保护数值安全
