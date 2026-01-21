# Reduce Operations Bang Backend Core Implementation Documentation

该模块实现了针对寒武纪 MLU（Cambricon MLU）硬件后端的高性能规约操作（Reduce Operations），包括求和（Sum）、平方和（Sum of Squares）和最大值（Max）等核心规约算法。所有函数均设计为 MLU 核函数（__mlu_func__），利用 NRAM（Neural RAM）和 BANG API 实现向量化加速，支持 float32、float16（half）和 bfloat16 三种数据类型。

## 1. Module Structure

- **`reduce_bang.h`**: 头文件实现，包含所有规约操作的 MLU 核函数模板实现。该文件定义了四种核心规约算法的基础实现和批处理优化版本。

## 2. Core Classes

该模块采用函数式编程范式，无类定义，主要包含以下模板函数组：

### `sumInternal`
- **Location**: `reduce_bang.h:10-28`
- **Primary Function**: 实现基础的浮点数求和规约，针对单批次数据执行向量化求和
- **Key Parameters**:
  - `float *dst`: 输出目标指针，单元素存储结果
  - `float *src`: 输入源数据指针（位于 NRAM）
  - `int max_batch`: 当前批次元素数量
- **Algorithm**:
  - 使用两阶段向量化规约策略：
    1. 当 `width >= 4` 时（数据量充足）：调用 `__bang_sumpool` 执行并行归约求和，然后通过 `__bang_reduce_sum` 对通道维度进行最终规约
    2. 当 `width < 4` 时（小数据量）：回退到串行 for 循环累加，避免向量化开销
- **Vectorization**: 128 字节对齐（32 个 float32 元素），通过 `batch_size = 128 / sizeof(float)` 定义

### `sumTyped<T>`
- **Location**: `reduce_bang.h:30-41`
- **Primary Function**: 类型感知的求和入口，根据输入数据类型（float32/half/bfloat16）自动执行类型转换并调用 `sumInternal`
- **Type Handling**:
  - `half`: 使用 `__bang_half2float` 进行 in-place 半精度转单精度
  - `bfloat16_t`: 使用 `__bang_bfloat162float` 进行 bfloat16 转 float32
  - `float`: 直接调用 `sumInternal`，无需转换
- **Memory Pattern**: 转换操作为 in-place（原地操作），复用输入缓冲区节省 NRAM 空间

### `sum<T>`
- **Location**: `reduce_bang.h:43-63`
- **Primary Function**: 完整的求和规约核函数，处理从 GDRAM（全局显存）到 NRAM 的数据迁移和分块处理
- **Key Parameters**:
  - `const T *source`: GDRAM 中的输入源数据
  - `T *src`: NRAM 中预分配的临时缓冲区
  - `float *dst`: NRAM 中存储单次批处理结果的标量缓冲区
  - `int num_elements`: 总元素数量
  - `int max_batch`: 单次批处理的最大元素数（受 NRAM 容量限制）
- **Algorithm**:
  1. **分块处理循环**：使用 `while (processed < num_elements)` 遍历数据
  2. **零填充策略**：当 `curr_batch < max_batch` 时，通过 `__bang_write_zero` 清零 NRAM 缓冲区，避免读取脏数据
  3. **DMA 传输**：调用 `__memcpy(..., GDRAM2NRAM)` 将数据从全局显存搬运到片上 NRAM
  4. **类型转换与规约**：调用 `sumTyped` 执行类型转换和向量化求和
  5. **累加结果**：将每批次结果 `dst[0]` 累加到 `res`
- **Buffer Layout**:
  - 对于 16 位类型（half/bfloat16）：`src` 缓冲区布局为 `[conversion_space (max_batch)] [original_data (max_batch)]`，通过 `offset = max_batch` 分隔
  - 对于 32 位类型（float）：直接使用 `src[0:max_batch]`，offset 为 0
- **Complexity**: O(num_elements)，每批次处理 O(max_batch)

### `sumBatched<T>`
- **Location**: `reduce_bang.h:65-111`
- **Primary Function**: 优化的批处理求和，针对大向量实现对齐优化和尾部处理
- **Optimization Strategy**:
  1. **小向量短路**：当 `num_elements < 32` 时，回退到 `sum` 函数，避免对齐优化的额外开销
  2. **对齐处理**：
     - `aligned_batch = (curr_batch / batch_size) * batch_size`：计算对齐到 128 字节边界的数据量
     - `remainder = curr_batch % batch_size`：计算剩余未对齐元素
  3. **分离处理**：
     - 对齐部分：调用 `sumInternal` 利用向量化指令
     - 未对齐尾部：使用串行 for 循环逐个累加
- **Performance Gain**: 消除非对齐访问的惩罚，提升内存访问效率
- **Memory Safety**: 每批次开始时强制 `__bang_write_zero` 清零，确保尾部数据不污染计算

### `sumSquared<T>`
- **Location**: `reduce_bang.h:113-147`
- **Primary Function**: 计算元素的平方和（用于方差、L2 范数等统计量）
- **Algorithm**:
  1. 分块搬运数据到 NRAM（同 `sum`）
  2. **逐元素平方**：使用 for 循环对每个元素执行 `val * val` 操作
  3. **类型转换策略**：
     - `half`: 使用 `__half2float` 内置函数转换后平方
     - `bfloat16_t`: 使用 `__bfloat162float` 内置函数转换后平方
     - `float`: 直接平方
  4. 累加平方值到 `res`
- **Use Cases**: 方差计算 `variance = sum_sq / n - mean^2`、L2 范数 `||x||_2 = sqrt(sum_sq)`

### `sumSquaredBatched<T>`
- **Location**: `reduce_bang.h:149-199`
- **Primary Function**: 向量化的平方和计算，利用 BANG 乘法指令加速
- **Optimization**:
  1. **向量化平方**：使用 `__bang_mul` 对整个对齐数组执行原地平方操作 `(float *)(src + offset) * (float *)(src + offset)`
  2. **分离处理**：对齐部分用 `sumInternal` + `__bang_mul`，尾部用串行循环
  3. **小向量优化**：`num_elements < 32` 时回退到 `sumSquared`
- **Performance**: `__bang_mul` 相比逐元素乘法有显著吞吐量提升

### `maxInternal`
- **Location**: `reduce_bang.h:201-213`
- **Primary Function**: 浮点数最大值规约，基于 maxpool 操作实现并行归约
- **Algorithm**:
  1. 调用 `__bang_maxpool` 执行通道维度的最大值池化：
     - 输入形状：`[batch_size, 1, width]`
     - 卷积核：`[1, width]` 覆盖整个宽度
     - 输出形状：`[batch_size, 1, 1]`
  2. 调用 `__bang_argmax` 对 `batch_size` 个通道求最大值索引，并提取值到 `dst[0]`
- **Parallelism**: maxpool 操作利用 MLU 的并行计算单元

### `maxTyped<T>`
- **Location**: `reduce_bang.h:215-226`
- **Primary Function**: 类型感知的最大值规约，支持 half/bfloat16/float 输入
- **Type Conversion**: 与 `sumTyped` 相同，使用 in-place 转换策略

### `max<T>`
- **Location**: `reduce_bang.h:228-248`
- **Primary Function**: 完整的最大值规约核函数，跨批次累加最大值
- **Initialization**: `max_val = -INFINITY` 确保第一个元素能正确更新最大值
- **Reduction**: 每批次调用 `maxTyped`，然后用 `std::max(max_val, dst[0])` 更新全局最大值

### `maxBatched<T>`
- **Location**: `reduce_bang.h:250-277`
- **Primary Function**: 优化的批处理最大值计算
- **Note**: 实现与 `max` 几乎相同，保留了小向量短路逻辑（`num_elements < 32`），但当前未使用对齐优化

## 3. API Interface

```cpp
namespace op::common_bang::reduce_op {

// 基础浮点求和规约（内部函数）
__mlu_func__ void sumInternal(float *dst, float *src, int max_batch);
// dst: 输出标量缓冲区（NRAM）
// src: 输入向量（NRAM，已转换为 float）
// max_batch: 批次大小，必须是对齐到 batch_size 的值

// 类型感知求和（内部函数）
template <typename T>
__mlu_func__ void sumTyped(float *result, T *data, size_t len);
// result: 输出标量指针
// data: 输入数据（支持 half/bfloat16_t/float）
// len: 数据长度

// 完整求和核函数（供外部调用）
template <typename T>
__mlu_func__ float sum(const T *source, T *src, float *dst,
                       int num_elements, int max_batch);
// source: GDRAM 输入数据（只读）
// src: NRAM 临时缓冲区（需预分配大小 max_batch * 2）
// dst: NRAM 单元素结果缓冲区
// num_elements: 总元素数
// max_batch: 单批处理最大元素数（建议 128/sizeof(float) 的倍数）
// 返回值: 所有元素的求和结果

// 向量化求和核函数（推荐用于大向量）
template <typename T>
__mlu_func__ float sumBatched(const T *source, T *src, float *dst,
                              int num_elements, int max_batch);
// 参数同 sum，内部使用对齐优化

// 平方和核函数
template <typename T>
__mlu_func__ float sumSquared(const T *source, T *src, float *dst,
                              int num_elements, int max_batch);
// 返回值: 所有元素的平方和

// 向量化平方和核函数
template <typename T>
__mlu_func__ float sumSquaredBatched(const T *source, T *src, float *dst,
                                     int num_elements, int max_batch);

// 基础浮点最大值规约（内部函数）
__mlu_func__ void maxInternal(float *dst, float *src, int max_batch);
// dst: 输出标量缓冲区
// src: 输入向量（NRAM，已转换为 float）

// 类型感知最大值（内部函数）
template <typename T>
__mlu_func__ void maxTyped(float *result, T *data, size_t len);

// 完整最大值核函数
template <typename T>
__mlu_func__ float max(const T *source, T *src, float *dst,
                       int num_elements, int max_batch);
// 返回值: 所有元素中的最大值

// 批处理最大值核函数
template <typename T>
__mlu_func__ float maxBatched(const T *source, T *src, float *dst,
                              int num_elements, int max_batch);

} // namespace op::common_bang::reduce_op
```

## 4. Usage Example

```cpp
#include "infiniop/reduce/bang/reduce_bang.h"

using namespace op::common_bang::reduce_op;

// 示例：在 MLU 核函数中计算 half 类型张量的求和、平方和与最大值
__mlu_func__ void reduce_kernel(const half *input_gdram, int num_elements,
                                float *sum_result, float *sum_sq_result,
                                float *max_result) {
    // 分配 NRAM 临时缓冲区
    // 对于 half 类型，需要 max_batch * 2 大小（原数据 + 转换空间）
    constexpr int max_batch = 1024;  // 单批处理 1024 个元素
    half nram_buffer[max_batch * 2]; // 类型 T 的缓冲区
    float nram_result;               // 单元素结果缓冲区

    // 1. 计算求和（使用向量化版本）
    float total_sum = sumBatched(input_gdram, nram_buffer, &nram_result,
                                 num_elements, max_batch);

    // 2. 计算平方和（使用向量化版本）
    float total_sum_sq = sumSquaredBatched(input_gdram, nram_buffer, &nram_result,
                                           num_elements, max_batch);

    // 3. 计算最大值
    float max_val = maxBatched(input_gdram, nram_buffer, &nram_result,
                               num_elements, max_batch);

    // 4. 将结果写回 GDRAM（假设有输出指针）
    // 实际应用中可能通过 __memcpy 写回全局内存
    *sum_result = total_sum;
    *sum_sq_result = total_sum_sq;
    *max_result = max_val;
}

// 示例：在主机端调用上述核函数
void host_launch_reduce(const half* d_input, int num_elements,
                        float* d_sum, float* d_sum_sq, float* d_max,
                        cnrtQueue_t queue) {
    // 使用 CNRT 启动核函数
    // 任务维度：1 个任务，1 个簇
    cnrtDim3_t dim = {1, 1, 1};
    cnrtFunction_t func;

    // 获取核函数并启动
    // （实际代码需包含 CNRT 的核函数注册和启动逻辑）
    cnrtInvokeKernel(func, dim, nullptr,
                     (void**)&d_input, (void**)&num_elements,
                     (void**)&d_sum, (void**)&d_sum_sq, (void**)&d_max,
                     nullptr);
}
```

## 5. Implementation Details

### Memory Management
- **NRAM 策略**：所有临时计算均在 NRAM（Neural RAM）中完成，这是 MLU 的片上高速存储器
- **Buffer 重用**：`src` 缓冲区在 half/bfloat16 场景下分区使用，前 `max_batch` 元素用于 float 转换结果，后 `max_batch` 元素存储原始数据，通过 `offset = max_batch` 分隔
- **零填充机制**：每批次处理前调用 `__bang_write_zero` 清空 NRAM，避免尾部数据污染（尤其是非对齐或最后一批次）
- **GDRAM-NRAM 传输**：使用 `__memcpy(..., GDRAM2NRAM)` 进行 DMA 传输，隐式同步

### Concurrency
- **SIMT 执行模型**：所有函数标记为 `__mlu_func__`，表示 MLU 核函数，运行在 MLU 的计算集群上
- **向量化并行**：`__bang_sumpool`、`__bang_maxpool`、`__bang_reduce_sum` 等指令利用 MLU 的并行计算单元，同时对多个元素执行规约
- **批处理并行**：虽然单个核函数内部是串行分批处理，但可通过 CNRT 启动多个任务并行处理不同数据块

### Performance
- **对齐优化**：`sumBatched`/`sumSquaredBatched` 系列函数将对齐数据（128 字节边界）与未对齐尾部分离处理，前者使用向量化指令，后者用标量循环
  - 对齐粒度：`batch_size = 128 / sizeof(float) = 32` 个 float 元素
  - 对齐收益：消除内存访问的非对齐惩罚，提升带宽利用率
- **算法选择**：
  - 小数据（`width < 4` 或 `num_elements < 32`）：回退到标量循环，避免向量化启动开销
  - 大数据：使用 `__bang_sumpool`（两阶段归约）或 `__bang_mul` + `__bang_reduce_sum`
- **复杂度保证**：
  - 所有算法均为 O(n) 时间复杂度
  - 空间复杂度：O(max_batch)，max_batch 为常数（受 NRAM 大小限制）

### Error Handling
- **无显式错误处理**：作为核函数，依赖 MLU 硬件的隐式错误检测（如地址越界会触发硬件异常）
- **类型安全**：使用 `if constexpr (std::is_same_v<T, ...>)` 编译期类型分发，避免运行时分支

### Dependencies
- **BANG API**：寒武纪 MLU 的基础算子库
  - `__bang_sumpool`：并行求和池化
  - `__bang_reduce_sum`：通道维规约求和
  - `__bang_maxpool`：最大值池化
  - `__bang_argmax`：最大值索引提取
  - `__bang_write_zero`：缓冲区清零
  - `__bang_mul`：向量乘法
  - `__bang_half2float` / `__bang_bfloat162float`：类型转换
- **CNRT**：寒武纪运行时 API
  - `__memcpy(..., GDRAM2NRAM)`：内存拷贝
- **标准库**：`<type_traits>`（用于 `std::is_same_v`）、`<cmath>`（用于 `INFINITY`）

### Design Patterns
- **Template Method Pattern**：`sumTyped`/`maxTyped` 定义算法骨架，类型转换通过 `if constexpr` 编译期特化
- **Strategy Pattern**：根据数据大小（`width >= 4`、`num_elements < 32`）选择向量化或标量策略
- **CRTP（Curiously Recurring Template Pattern）**：虽然未显式使用，但模板函数的重载机制实现了类似效果
- **Buffer Reuse Pattern**：通过 offset 机制实现同一缓冲区的多用途复用（转换空间 + 原始数据）

### Hardware-Specific Optimizations
- **NRAM 容量限制**：`max_batch` 参数受 NRAM 大小约束（参考 `common_bang.h` 中 `NRAM_MAX_SIZE = 1024 * 240`），典型值不超过数百 KB
- **128 字节对齐**：`batch_size = 128 / sizeof(float)` 确保内存访问对齐到 MLU 的缓存行大小
- **in-place 转换**：half/bfloat16 转 float 直接在原缓冲区执行，减少内存拷贝
- **向量化指令**：`__bang_*` 系列函数充分利用 MLU 的 SIMD/MIMD 并行能力
