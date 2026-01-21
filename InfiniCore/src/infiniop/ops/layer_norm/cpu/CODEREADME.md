# Layer Norm CPU 实现文档 (Layer Norm CPU Implementation Documentation)

## 模块概述 (Module Overview)

该模块实现了 Layer Normalization 层归一化操作的 CPU 版本后端。Layer Normalization 是深度学习中常用的归一化技术,通过对每个样本的所有特征进行归一化来稳定训练过程。本实现支持 FP16、BF16 和 FP32 三种数据类型,使用 OpenMP 进行并行计算优化。

核心算法实现包括:均值计算、方差计算、标准化输出、以及可选的仿射变换(affine transformation)。

## 1. 模块结构 (Module Structure)

- **`layer_norm_cpu.h`**: CPU 后端描述符声明头文件,使用宏定义生成 Descriptor 类框架
- **`layer_norm_cpu.cc`**: CPU 后端核心实现,包含模板化的 layer_norm 计算函数和描述符接口实现

## 2. 核心类与数据结构 (Core Classes and Data Structures)

### `op::layer_norm::LayerNormInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/layer_norm/info.h`
- **主要功能**: 封装 Layer Normalization 操作的所有元数据和配置信息
- **关键字段**:
  - `infiniDtype_t dtype`: 输入/输出的数据类型 (F16/F32/BF16)
  - `size_t ndim`: 张量的维度数
  - `std::vector<size_t> input_shape`: 输入张量的形状
  - `size_t normalized_size`: 最后一个维度的大小(归一化维度)
  - `size_t othersize`: 除最后一维外所有维度的乘积
  - `std::vector<ptrdiff_t> output_strides`: 输出张量的步长(stride)数组
  - `std::vector<ptrdiff_t> input_standardization_strides`: 标准化输出张量的步长
  - `std::vector<ptrdiff_t> input_std_deviation_strides`: 标准差输出张量的步长
  - `std::vector<ptrdiff_t> input_strides`: 输入张量的步长
  - `std::vector<ptrdiff_t> weight_strides`: 权重张量的步长
  - `std::vector<ptrdiff_t> bias_strides`: 偏置张量的步长
  - `float eps`: 数值稳定性的小常数,防止除零
  - `bool bias_exist`: 是否存在偏置项

**核心方法**:
- `static Result<LayerNormInfo> createLayerNormInfo(...)`: 静态工厂方法,验证输入张量描述符并构建 LayerNormInfo 对象

**验证逻辑**:
1. 检查 output、input、input_standardization 三个张量形状完全一致
2. weight 必须是 1 维向量,长度等于最后一维大小
3. bias (可选)必须是 1 维向量,长度等于最后一维大小
4. input_std_deviation 的维度数比输入少 1,且前 ndim-1 维与输入相同

### `op::layer_norm::cpu::Descriptor`
- **位置**: `layer_norm_cpu.cc` (通过宏定义生成)
- **主要功能**: Layer Norm CPU 后端的操作描述符,继承自 `InfiniopDescriptor`
- **继承结构**:
  ```
  InfiniopDescriptor
      └── Descriptor (op::layer_norm::cpu)
  ```
- **关键字段**:
  - `struct Opaque *_opaque`: 不透明指针(本实现中未使用,设为 nullptr)
  - `LayerNormInfo _info`: 层归一化的配置信息对象
  - `size_t _workspace_size`: 工作空间大小(当前实现为 0,无需额外内存)
- **生命周期**:
  1. **构造**: 通过 `Descriptor::create()` 静态方法创建,验证参数并初始化
  2. **使用**: 调用 `calculate()` 方法执行计算
  3. **析构**: 默认析构函数释放资源

**核心方法**:
- `static infiniStatus_t create(...)`: 静态工厂方法,创建描述符实例
  - 参数: handle、输出和输入张量描述符、eps 值
  - 返回: `INFINI_STATUS_SUCCESS` 或错误码
  - 验证数据类型必须是 F16/F32/BF16 之一

- `infiniStatus_t calculate(...)`: 执行层归一化计算
  - 参数: workspace、输出/输入指针、weight/bias 指针、stream(CPU 后端忽略)
  - 实现方式: 根据 dtype 分发到对应的模板特化版本
  - 返回: `INFINI_STATUS_SUCCESS` 或错误码

## 3. 核心算法实现 (Core Algorithm Implementation)

### `calculate_layer_norm<Tdata>()`
- **位置**: `layer_norm_cpu.cc` 第 8-47 行
- **函数签名**:
  ```cpp
  template <typename Tdata>
  infiniStatus_t calculate_layer_norm(
      const LayerNormInfo &info,
      Tdata *output,
      Tdata *input_standardization,
      Tdata *input_std_deviation,
      const Tdata *input,
      const Tdata *weight,
      const Tdata *bias)
  ```
- **算法描述**: 对输入张量执行 Layer Normalization

**详细流程**:

1. **并行化策略** (第 18 行):
   ```cpp
   #pragma omp parallel for
   for (int b = 0; b < (int)(info.input_shape[0] * info.input_shape[1]); b++)
   ```
   - 使用 OpenMP 并行化外层循环
   - 并行粒度: 批次维度 × 第二维度 (shape[0] × shape[1])
   - 每个线程独立处理一个样本的所有特征维度

2. **指针计算** (第 19-24 行):
   - 通过除法/取模将扁平索引 `b` 分解为 `(b0, b1)` 二维索引
   - 使用 stride 数组计算各张量的起始指针偏移
   - 支持非连续内存布局(tensors with strides)

3. **均值计算** (第 25-29 行):
   ```cpp
   float mean = op::common_cpu::reduce_op::sum(
                    input_ptr,
                    info.normalized_size,
                    info.input_strides[2])
              / info.input_shape[2];
   ```
   - 调用 `reduce_op::sum()` 对最后一维求和
   - 除以归一化大小得到算术平均值
   - 时间复杂度: O(normalized_size)

4. **方差计算** (第 30-34 行):
   ```cpp
   float sum_sq = op::common_cpu::reduce_op::sumSquared(
       input_ptr,
       info.normalized_size,
       info.input_strides[2]);
   float var = sum_sq / (info.normalized_size) - mean * mean;
   ```
   - 调用 `reduce_op::sumSquared()` 计算平方和
   - 使用公式 `var = E[x²] - (E[x])²` 计算方差
   - 数值稳定性: 可能出现微小负数,但后续 +eps 会修正

5. **标准差计算** (第 35-36 行):
   ```cpp
   float std_deviation = std::sqrt(var + info.eps);
   *std_ptr = utils::cast<Tdata>(std_deviation);
   ```
   - 添加 eps 防止 sqrt(0) 或 sqrt(负数)
   - 将标准差写入 `input_std_deviation` 输出张量
   - 使用 `utils::cast` 进行类型转换(FP32 → FP16/BF16/F32)

6. **标准化与仿射变换** (第 38-43 行):
   ```cpp
   for (size_t d = 0; d < info.normalized_size; d++) {
       float x_standard = (utils::cast<float>(*(input_ptr + d * info.input_strides[2])) - mean) / std_deviation;
       *(standard_ptr + d * info.input_standardization_strides[2]) = utils::cast<Tdata>(x_standard);
       *(output_ptr + d * info.output_strides[2]) = utils::cast<Tdata>(
           x_standard * utils::cast<float>(*(weight + d * info.weight_strides[0])) +
           (info.bias_exist ? utils::cast<float>(*(bias + d * info.bias_strides[0])) : float(0)));
   }
   ```
   - **标准化**: `x_standard = (x - mean) / std_deviation`
   - **写入标准化输出**: 存储到 `input_standardization` 张量
   - **仿射变换**: `output = x_standard * weight + bias` (可选)
   - 类型安全: 所有计算在 FP32 中进行,最后转换为输出类型

**并行性能分析**:
- OpenMP 线程级并行,粒度为样本维度
- 每个线程独立完成完整的 layer_norm 流程
- 无共享状态,无线程同步开销
- 内存访问模式: 每个线程访问独立的样本数据,cache-friendly

## 4. API 接口 (API Interface)

### 描述符创建接口
```cpp
infiniStatus_t op::layer_norm::cpu::Descriptor::create(
    infiniopHandle_t handle_,                          // [in] CPU 设备句柄
    Descriptor **desc_ptr,                             // [out] 输出的描述符指针
    infiniopTensorDescriptor_t output_desc,            // [in] 输出张量描述符
    infiniopTensorDescriptor_t input_standardization_desc, // [in] 标准化输出描述符
    infiniopTensorDescriptor_t input_std_deviation_desc,   // [in] 标准差输出描述符
    infiniopTensorDescriptor_t input_desc,             // [in] 输入张量描述符
    infiniopTensorDescriptor_t weight_desc,            // [in] 权重张量描述符
    infiniopTensorDescriptor_t bias_desc,              // [in] 偏置张量描述符(可为 null)
    float eps);                                        // [in] 数值稳定性参数
```
- **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`,失败返回对应错误码
- **错误检查**:
  - `CHECK_DTYPE`: 验证数据类型为 F16/F32/BF16
  - `LayerNormInfo::createLayerNormInfo`: 验证张量形状一致性

### 计算执行接口
```cpp
infiniStatus_t op::layer_norm::cpu::Descriptor::calculate(
    void *workspace,          // [in] 工作空间指针(当前未使用)
    size_t workspace_size,    // [in] 工作空间大小(当前为 0)
    void *output,             // [out] 输出数据指针
    void *input_standardization, // [out] 标准化输出数据指针
    void *input_std_deviation,   // [out] 标准差输出数据指针
    const void *input,        // [in] 输入数据指针
    const void *weight,       // [in] 权重数据指针
    const void *bias,         // [in] 偏置数据指针(可为 null)
    void *stream) const;      // [in] 流指针(CPU 后端忽略)
```
- **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`,失败返回错误码
- **类型分发逻辑**:
  ```cpp
  if (_info.dtype == INFINI_DTYPE_F16) {
      CALCULATE_LAYER_NORM(fp16_t);
  } else if (_info.dtype == INFINI_DTYPE_BF16) {
      CALCULATE_LAYER_NORM(bf16_t);
  } else if (_info.dtype == INFINI_DTYPE_F32) {
      CALCULATE_LAYER_NORM(float);
  }
  ```
  使用宏 `CALCULATE_LAYER_NORM(TDATA)` 实例化模板函数

## 5. 使用示例 (Usage Example)

```cpp
#include "infiniop/operator.h"
#include "infiniop/tensor.h"
#include "infiniop/ops/layer_norm/cpu/layer_norm_cpu.h"

// 示例: 对 [batch_size, seq_len, hidden_dim] 的张量执行 layer norm
void example_layer_norm() {
    // 1. 配置参数
    const size_t batch_size = 32;
    const size_t seq_len = 128;
    const size_t hidden_dim = 768;
    const float eps = 1e-5f;

    // 2. 创建张量描述符
    // 输入张量: [batch_size, seq_len, hidden_dim]
    size_t input_shape[] = {batch_size, seq_len, hidden_dim};
    size_t input_strides[] = {seq_len * hidden_dim, hidden_dim, 1};
    infiniopTensorDescriptor_t input_desc;
    infiniopCreateTensorDescriptor(&input_desc,
        INFINI_DTYPE_F32,           // 数据类型
        3,                          // 维度数
        input_shape,
        input_strides,
        0);                         // 偏移量

    // 输出张量: 与输入形状相同
    infiniopTensorDescriptor_t output_desc;
    infiniopCreateTensorDescriptor(&output_desc,
        INFINI_DTYPE_F32, 3, input_shape, input_strides, 0);

    // 标准化输出张量: 与输入形状相同
    infiniopTensorDescriptor_t input_standardization_desc;
    infiniopCreateTensorDescriptor(&input_standardization_desc,
        INFINI_DTYPE_F32, 3, input_shape, input_strides, 0);

    // 标准差输出张量: [batch_size, seq_len]
    size_t std_shape[] = {batch_size, seq_len};
    size_t std_strides[] = {seq_len, 1};
    infiniopTensorDescriptor_t input_std_deviation_desc;
    infiniopCreateTensorDescriptor(&input_std_deviation_desc,
        INFINI_DTYPE_F32, 2, std_shape, std_strides, 0);

    // 权重张量: [hidden_dim]
    size_t weight_shape[] = {hidden_dim};
    size_t weight_strides[] = {1};
    infiniopTensorDescriptor_t weight_desc;
    infiniopCreateTensorDescriptor(&weight_desc,
        INFINI_DTYPE_F32, 1, weight_shape, weight_strides, 0);

    // 偏置张量: [hidden_dim] (可选)
    infiniopTensorDescriptor_t bias_desc;
    infiniopCreateTensorDescriptor(&bias_desc,
        INFINI_DTYPE_F32, 1, weight_shape, weight_strides, 0);

    // 3. 获取 CPU 句柄
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

    // 4. 创建 Layer Norm 描述符
    op::layer_norm::cpu::Descriptor *layer_norm_desc;
    infiniStatus_t status = op::layer_norm::cpu::Descriptor::create(
        handle,
        &layer_norm_desc,
        output_desc,
        input_standardization_desc,
        input_std_deviation_desc,
        input_desc,
        weight_desc,
        bias_desc,
        eps);

    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理
        return;
    }

    // 5. 分配并初始化内存
    size_t total_elements = batch_size * seq_len * hidden_dim;
    float *input_data = new float[total_elements];
    float *output_data = new float[total_elements];
    float *input_standardization_data = new float[total_elements];
    float *input_std_deviation_data = new float[batch_size * seq_len];
    float *weight_data = new float[hidden_dim];
    float *bias_data = new float[hidden_dim];

    // 初始化输入、权重、偏置数据
    std::fill_n(input_data, total_elements, 1.0f);
    std::fill_n(weight_data, hidden_dim, 1.0f);
    std::fill_n(bias_data, hidden_dim, 0.0f);

    // 6. 执行 Layer Norm 计算
    status = layer_norm_desc->calculate(
        nullptr,                          // workspace (当前不需要)
        0,                                // workspace_size
        output_data,                      // 输出
        input_standardization_data,       // 标准化输出
        input_std_deviation_data,         // 标准差
        input_data,                       // 输入
        weight_data,                      // 权重
        bias_data,                        // 偏置
        nullptr);                         // stream (CPU 后端忽略)

    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理
        delete[] input_data;
        delete[] output_data;
        delete[] layer_norm_desc;
        return;
    }

    // 7. 验证结果 (可选)
    // 对于全 1 输入,标准化后应为 0,输出应为 bias(0)
    for (size_t i = 0; i < total_elements; i++) {
        assert(fabs(output_data[i]) < 1e-6f);
    }

    // 8. 清理资源
    delete[] input_data;
    delete[] output_data;
    delete[] input_standardization_data;
    delete[] input_std_deviation_data;
    delete[] weight_data;
    delete[] bias_data;
    delete layer_norm_desc;

    infiniopDestroyTensorDescriptor(input_desc);
    infiniopDestroyTensorDescriptor(output_desc);
    infiniopDestroyTensorDescriptor(input_standardization_desc);
    infiniopDestroyTensorDescriptor(input_std_deviation_desc);
    infiniopDestroyTensorDescriptor(weight_desc);
    infiniopDestroyTensorDescriptor(bias_desc);
    infiniopDestroyHandle(handle);
}
```

## 6. 实现细节 (Implementation Details)

### 内存管理 (Memory Management)
- **工作空间策略**: 当前实现不需要额外工作空间,`workspace_size = 0`
- **零拷贝**: 所有计算直接在输入/输出指针上进行,无中间缓冲区分配
- **类型安全**: 通过 `utils::cast<Tdata>()` 确保类型转换的正确性
  - FP16/BF16 ↔ FP32 转换使用专用转换函数
  - 避免精度损失和溢出问题

### 并发控制 (Concurrency)
- **并行框架**: OpenMP (`#pragma omp parallel for`)
- **并行粒度**: 批次维度 × 序列维度 (shape[0] × shape[1])
- **数据竞争避免**: 每个线程处理独立的样本,无共享可变状态
- **同步开销**: 零同步开销,线程完全独立执行
- **NUMA 优化**: 可通过 OMP 环境变量控制线程亲和性和 NUMA 策略

### 性能优化 (Performance Optimizations)
1. **循环展开**: 编译器可自动展开内层循环 (归一化维度循环)
2. **向量化潜力**: 内层循环适合 SIMD 向量化 (连续内存访问)
   - GCC/Clang 可自动向量化 FP32 操作
   - FP16/BF16 需先转换为 FP32 再计算
3. **Cache 友好**:
   - 每个线程访问连续内存块
   - stride 访问模式支持张量的非连续布局
4. **计算强度**:
   - 每个元素执行约 10 次浮点运算 (sum, sum_sq, mean, var, sqrt, div, mul, add)
   - 内存访问: 读 2 次(input, weight),写 2 次(output, standardization)

### 数值稳定性 (Numerical Stability)
- **eps 参数**: 防止标准差为 0 时的除零错误
  - `std_deviation = sqrt(var + eps)`
  - 典型值: 1e-5 或 1e-6
- **方差计算**: 使用 `E[x²] - (E[x])²` 而非 `E[(x-μ)²]`
  - 优势: 两次遍历即可完成,避免存储中间残差
  - 缺点: 可能出现微小负数 (catastrophic cancellation)
  - 缓解: 添加 eps 使其非负
- **精度保持**: 所有中间计算在 FP32 中进行
  - FP16/BF16 输入立即转换为 FP32
  - 避免累积舍入误差

### 错误处理 (Error Handling)
- **错误传播**: 使用 `CHECK_DTYPE` 和 `CHECK_RESULT` 宏进行参数验证
- **错误码**:
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 数据类型不支持
  - `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量形状不匹配
- **资源管理**:
  - 创建失败时不会分配 Descriptor,无需清理
  - 析构函数默认实现,无动态资源释放

### 依赖关系 (Dependencies)
1. **外部依赖**:
   - OpenMP (可选,`ENABLE_OMP` 宏控制)
   - C++ 标准库: `<cmath>`, `<cstddef>`, `<vector>`

2. **内部依赖**:
   - `op::common_cpu::reduce_op::sum()`: 归约求和操作
   - `op::common_cpu::reduce_op::sumSquared()`: 平方和归约操作
   - `utils::cast<T>()`: 类型转换工具
   - `LayerNormInfo`: 元数据容器
   - `InfiniopDescriptor`: 基类

3. **模块关系**:
   ```
   layer_norm/cpu
       ├── reduce/cpu (归约操作)
       ├── devices/cpu (CPU 设备抽象)
       └── utils (类型转换和工具函数)
   ```

### 设计模式 (Design Patterns)
1. **策略模式 (Strategy Pattern)**:
   - 不同后端 (CPU, CUDA, Kunlun 等) 实现 Descriptor 接口
   - 运行时通过设备类型选择策略

2. **模板方法模式 (Template Method Pattern)**:
   - `Descriptor::create()` 定义创建流程
   - 子类提供具体实现细节

3. **工厂模式 (Factory Pattern)**:
   - `Descriptor::create()` 作为静态工厂方法
   - 封装对象创建逻辑和参数验证

4. **RAII (Resource Acquisition Is Initialization)**:
   - Descriptor 对象封装所有配置信息
   - 析构时自动释放资源

### 支持的数据类型 (Supported Data Types)
- **FP16 (半精度浮点)**: `fp16_t` → 计算时转为 FP32
- **BF16 (脑浮点数)**: `bf16_t` → 计算时转为 FP32
- **FP32 (单精度浮点)**: `float` → 直接计算
- **限制**: 不支持整型、双精度等其他类型

### 张量布局支持 (Tensor Layout Support)
- **连续内存**: 标准 C-order (row-major) 布局
- **非连续张量**: 通过 stride 数组支持任意布局
  - 例如: 转置张量、切片张量、广播张量
- **广播机制**: 当前实现不支持广播,输入/输出形状必须一致

## 7. 复杂度分析 (Complexity Analysis)

### 时间复杂度 (Time Complexity)
- **单样本**: O(normalized_size)
  - 均值计算: O(d) 其中 d = normalized_size
  - 方差计算: O(d)
  - 标准化: O(d)
  - 总计: O(3d) = O(d)

- **整体**: O(batch × seq × normalized_dim)
  - 等价于 O(总元素数)
  - 线性复杂度,无法避免(必须访问每个元素)

### 空间复杂度 (Space Complexity)
- **额外空间**: O(1)
  - 仅使用局部变量 (mean, var, std_deviation)
  - 无中间缓冲区分配
  - 工作空间大小为 0

- **输入输出空间**: O(n) 其中 n 为总元素数
  - 输入、输出、标准化输出、标准差输出、权重、偏置

### 并行扩展性 (Parallel Scalability)
- **理论加速比**: 接近 CPU 核心数
  - 任务完全独立,无同步开销
  - 适合多核 CPU 并行
- **实际限制**:
  - 内存带宽瓶颈: 大 batch 时可能受限于内存访问速度
  - cache 容量: 小 batch 时可能无法充分利用所有核心

## 8. 局限性与未来改进 (Limitations and Future Improvements)

### 当前局限性
1. **仅支持最后维度归一化**: 硬编码为对最后一维 (dim=ndim-1) 进行归一化
2. **无广播支持**: 输入/输出形状必须完全一致
3. **无原地计算**: 不支持 in-place 操作 (输入输出共用内存)
4. **固定并行策略**: OpenMP 粒度固定,无法根据数据大小动态调整

### 可能的优化方向
1. **SIMD 向量化**: 显式使用 AVX2/AVX-512 指令加速归约和标准化
2. **Cache 分块**: 对大张量进行分块,提高 cache 命中率
3. **混合精度**: 使用低精度累加器 (如 FP16 累加) 减少内存流量
4. **多线程归约**: 在归约操作内部也使用 OpenMP 并行
5. **支持更多数据类型**: 如 INT8 量化输入的 layer norm
