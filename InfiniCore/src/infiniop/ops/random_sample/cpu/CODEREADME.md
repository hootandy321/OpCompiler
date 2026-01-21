# Random Sample CPU 操作实现文档

本模块实现了在 CPU 设备上进行随机采样的核心功能，广泛应用于大语言模型（LLM）的文本生成场景。该模块支持基于概率分布的采样策略，包括 top-k、top-p（nucleus sampling）、温度缩放等关键参数控制。

## 1. 模块结构

- **`random_sample_cpu.h`**: CPU 设备描述符声明文件，通过宏 `DESCRIPTOR(cpu)` 展开生成完整的 Descriptor 类定义
- **`random_sample_cpu.cc`**: CPU 后端核心实现，包含描述符创建、工作空间管理、采样算法的具体实现

## 2. 核心类

### `op::random_sample::cpu::Descriptor`
- **位置**: `random_sample_cpu.h` (通过宏展开), `random_sample_cpu.cc`
- **主要功能**: 封装 CPU 设备上的随机采样操作描述符，管理算子生命周期和执行调度
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **关键成员**:
  - `_info`: `RandomSampleInfo` 结构体，存储输入/输出张量的类型和形状信息（索引类型 dt_i、概率类型 dt_p、概率分布长度 n）
  - `_min_workspace_size`: `size_t` 类型，工作空间最小需求（当前固定为 0）
  - `_opaque`: 不透明指针，预留用于未来扩展（当前构造时传入 nullptr）
- **核心方法**:
  - `create(handle_, desc_ptr, result_desc, probs_desc)`: 静态工厂方法，验证张量描述符并构造描述符实例。使用 `RandomSampleInfo::create()` 进行形状和类型校验，将句柄转换为 `device::cpu::Handle*` 类型，通过 `new` 分配内存并返回
  - `minWorkspaceSize()`: 返回 `_min_workspace_size` 常量（0），表示当前实现不需要额外工作空间
  - `calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream)`: 执行采样计算的主入口。通过 `Calculate::calculate<Algo>()` 静态分发到模板化的算法实现，支持整数索引类型（i8/i16/i32/i64/u8/u16/u32/u64）和浮点概率类型（fp16/bf16/f32/f64）的组合
- **生命周期**: 通过 `create()` 静态方法构造（使用 `new` 分配），析构函数默认实现，由调用方负责释放

### `ComputeType<T>` 类型萃取模板
- **位置**: `random_sample_cpu.cc:34-46`
- **主要功能**: 定义计算精度类型映射策略
- **特化规则**:
  - 默认模板: `using type = T` （保持原类型）
  - `fp16_t` 特化: `using type = float` （半精度提升到单精度）
  - `bf16_t` 特化: `using type = float` （脑浮点提升到单精度）
- **设计目的**: 在低精度输入（fp16/bf16）时使用 float 进行数值计算以避免精度损失和溢出

### `Algo` 策略结构体
- **位置**: `random_sample_cpu.cc:48-117`
- **主要功能**: 封装采样算法的两种执行策略：确定性 argmax 和随机采样
- **核心方法**:

  - **`get<Tidx, Tval>(ptr, i)`**:
    - 读取概率数组的第 i 个元素并进行类型转换
    - 使用 `utils::cast<ComputeType<Tval>::type, Tval>()` 将存储类型（如 fp16）转换为计算类型（如 float）
    - 返回高精度的浮点值用于后续比较

  - **`argmax<Tidx, Tval>(workspace, workspace_size, result, probs, n, stream)`**:
    - **功能**: 确定性采样，直接选择概率最大的元素索引（贪婪解码）
    - **算法**: 单次遍历 O(n) 时间复杂度，维护当前最大值和对应索引
    - **触发条件**: `random_val == 0 || topp == 0 || topk == 1 || temperature == 0`
    - **实现细节**:
      1. 初始化 `idx = 0`, `max_val = probs[0]`
      2. 遍历 i 从 1 到 n-1
      3. 如果 `probs[i] > max_val`，更新 `max_val` 和 `idx`
      4. 将结果写入 `*reinterpret_cast<Tidx*>(result)`

  - **`random<Tidx, Tval>(workspace, workspace_size, result, probs, n, random_val, topp, topk, temperature, stream)`**:
    - **功能**: 随机采样，支持 top-k、top-p 和温度缩放
    - **算法**: 基于 softmax 的加权随机采样，时间复杂度 O(n log n)（主要由排序决定）
    - **数据结构**: `std::vector<KVPair>` 存储索引-概率对，其中 `KVPair::operator<` 重载为降序排列（`val > other.val`）
    - **实现步骤**:
      1. **构建索引-概率对** (O(n)): 创建 `pairs` 数组，每个元素包含 `{索引, 概率值}`
      2. **排序** (O(n log n)): 使用 `std::sort` 按概率降序排列
      3. **计算累积概率** (O(n)): 先对概率值应用温度缩放 softmax，计算公式为 `exp((val - max_val) / temperature)`，然后计算前缀和（累积分布函数 CDF）
      4. **确定采样边界**:
         - `pk`: 第 topk 个元素的累积概率（top-k 截断）
         - `pp`: 总和乘以 topp（top-p/nucleus 截断）
         - `plimit`: `random_val * min(pk, pp)`，实际采样阈值
      5. **采样** (O(n)): 遍历排序后的数组，找到第一个累积概率大于等于 `plimit` 的索引
    - **数值稳定性**:
      - 使用 `max_val` 作为基准值，避免 `exp()` 上溢
      - softmax 第一个元素设为 1（相当于 `exp(0)`），后续元素累加

## 3. API 接口

```cpp
namespace op::random_sample::cpu {

// 描述符创建接口
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // [in] Infini 运行时句柄
    Descriptor **desc_ptr,                 // [out] 输出的描述符指针
    infiniopTensorDescriptor_t result_desc,// [in] 结果张量描述符（0 维标量）
    infiniopTensorDescriptor_t probs_desc  // [in] 概率张量描述符（1 维向量）
);
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码
// 作用: 验证张量形状和类型，构造并初始化描述符对象

// 工作空间查询接口
size_t Descriptor::minWorkspaceSize() const;
// 返回: 所需工作空间最小字节数（当前实现返回 0）
// 作用: 允许调用方预分配内存，当前 CPU 实现不需要额外工作空间

// 核心计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,           // [in] 工作空间指针（可为 nullptr）
    size_t workspace_size,     // [in] 工作空间大小（字节）
    void *result,              // [out] 输出标量，存储采样到的索引
    const void *probs,         // [in] 输入概率分布数组
    float random_val,          // [in] 随机数 [0, 1]，用于控制采样位置
    float topp,                // [in] top-p 阈值 [0, 1]，控制累积概率截断
    int topk,                  // [in] top-k 值，限制只从前 k 个候选中采样
    float temperature,         // [in] 温度参数，控制分布平滑度（越大越均匀）
    void *stream               // [in] 流指针（CPU 后端未使用，保留接口兼容性）
) const;
// 返回: 成功返回 INFINI_STATUS_SUCCESS
// 作用: 执行采样计算，根据参数选择 argmax 或随机采样策略

}
```

## 4. 使用示例

```cpp
// 示例: CPU 设备上执行随机采样
#include "infiniop/ops/random_sample/cpu/random_sample_cpu.h"

// 1. 准备句柄和描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

// 2. 创建张量描述符
// 结果: 0 维标量（int32 类型）
int64_t result_shape[] = {};
infiniopTensorDescriptor_t result_desc;
infiniopCreateTensorDescriptor(&result_desc,
    INFINI_DTYPE_I32, 0, result_shape, nullptr);

// 概率: 1 维向量（float 类型，长度为 5）
int64_t probs_shape[] = {5};
int64_t probs_strides[] = {1};  // 必须紧凑存储
float probs_data[] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};
infiniopTensorDescriptor_t probs_desc;
infiniopCreateTensorDescriptor(&probs_desc,
    INFINI_DTYPE_F32, 1, probs_shape, probs_strides);

// 3. 创建操作描述符
op::random_sample::cpu::Descriptor *sample_desc;
auto status = op::random_sample::cpu::Descriptor::create(
    handle, &sample_desc, result_desc, probs_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（类型不匹配、形状错误等）
}

// 4. 准备输出和采样参数
int32_t result_idx;
float random_val = 0.73f;  // 假设从随机数生成器获得
float topp = 0.9f;         // nucleus sampling 阈值
int topk = 3;              // 只从前 3 个候选中选择
float temperature = 0.8f;  // 温度缩放

// 5. 执行采样（CPU 后端不需要工作空间和流）
status = sample_desc->calculate(
    nullptr, 0,              // workspace
    &result_idx, probs_data, // data pointers
    random_val, topp, topk, temperature,
    nullptr                  // stream (未使用)
);

// 6. 使用结果
printf("Sampled token index: %d\n", result_idx);  // 输出类似: "Sampled token index: 2"

// 7. 清理资源
delete sample_desc;
infiniopDestroyTensorDescriptor(result_desc);
infiniopDestroyTensorDescriptor(probs_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

- **内存管理**:
  - 描述符对象使用 C++ `new` 动态分配，由调用方负责 `delete` 释放
  - 不需要工作空间（`_min_workspace_size = 0`），所有计算使用栈内存或临时 `std::vector`
  - `std::vector<KVPair>` 在 `random()` 方法中临时分配，存储 n 个 `KVPair` 结构体（索引+概率值）

- **并发控制**:
  - 当前实现为单线程 CPU 代码，未使用 OpenMP 或其他并行化机制
  - 线程安全性由调用方保证（描述符对象不应被多线程同时调用）
  - 流参数（`stream`）保留用于未来异步执行支持，当前未使用

- **性能特征**:
  - **argmax 模式**: O(n) 时间复杂度，O(1) 空间复杂度，仅遍历一次数组
  - **random 模式**: O(n log n) 时间复杂度（瓶颈在 `std::sort`），O(n) 空间复杂度（存储 pairs 数组）
  - **数值计算**: 使用 `std::exp()` 进行 softmax 变换，低精度类型（fp16/bf16）自动提升到 float 计算
  - **排序策略**: 使用 `std::sort`（通常是内省排序 Introsort，最坏 O(n log n)），`KVPair::operator<` 重载为降序以配合 top-p/top-k 截断逻辑

- **错误处理**:
  - 描述符创建阶段通过 `RandomSampleInfo::create()` 进行严格校验：
    - 结果张量必须是整数类型（`CHECK_DTYPE_ANY_INT`）
    - 概率张量必须是浮点类型（f16/bf16/f32/f64）
    - 结果张量形状必须是 0 维标量（`ndim() == 0`）
    - 概率张量形状必须是 1 维向量（`ndim() == 1`）
    - 概率张量步幅必须为 1（紧凑存储，`stride(0) == 1`）
  - 校验失败返回对应的 `infiniStatus_t` 错误码（如 `INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - 计算过程中假设输入已通过校验，不进行额外检查（`switch` 语句的 `default` 分支调用 `std::abort()`）

- **类型分发策略**:
  - 使用 **双层模板元编程** 实现类型特化：
    1. 外层通过 `Calculate::calculate<Algo>()` 遍历索引类型（i8/i16/i32/i64/u8/u16/u32/u64）
    2. 内层通过 `switch_val()` 遍历概率类型（f16/bf16/f32/f64）
    3. 最终特化到具体的 `Algo::argmax<Tidx, Tval>()` 或 `Algo::random<Tidx, Tval>()`
  - 编译期生成所有类型组合的特化版本，避免运行时分支开销

- **设计模式**:
  - **策略模式 (Strategy Pattern)**: `Algo` 结构体封装了 `argmax` 和 `random` 两种算法，通过模板参数在编译期选择
  - **工厂模式 (Factory Pattern)**: `Descriptor::create()` 静态方法作为构造入口，封装对象创建逻辑
  - **类型萃取 (Type Traits)**: `ComputeType<T>` 模板定义类型映射规则，实现编译期类型转换
  - **CRTP (Curiously Recurring Template Pattern)**: `Calculate` 基类通过模板参数接收 `Algo` 派生类型，实现静态多态

- **依赖关系**:
  - **内部依赖**:
    - `../random_sample.h`: 定义描述符宏 `DESCRIPTOR` 和 `Calculate` 基类
    - `../info.h`: 定义 `RandomSampleInfo` 结构体和张量校验逻辑
    - `../../../devices/cpu/common_cpu.h`: CPU 设备公共头文件（当前未直接使用）
    - `../../../utils.h`: 工具函数（类型转换、错误检查宏如 `CHECK_RESULT`）
    - `"infinicore.h"`: 全局类型定义和枚举
  - **外部依赖**:
    - `<algorithm>`: 提供 `std::sort`, `std::min`
    - `<vector>`: 提供 `std::vector` 容器
    - `<cmath>`: 提供 `std::exp` 数学函数（通过间接包含）

- **数值稳定性保证**:
  - **softmax 计算**: 使用 `max_val` 作为基准，即 `exp((x - max_val) / temperature)`，避免 `exp(x)` 在 x 很大时溢出
  - **累积概率**: 第一个元素设为 1（相当于 `exp(0)`），后续累加，保证单调递增且数值稳定
  - **低精度提升**: fp16/bf16 自动转换为 float 计算，避免半精度浮点的精度损失和下溢

- **采样策略细节**:
  - **贪婪解码 (Argmax)**: 当 `random_val == 0 || topp == 0 || topk == 1 || temperature == 0` 时触发，直接选择最大概率索引，结果确定可复现
  - **top-k 采样**: 限制只在前 k 个最大概率元素中采样，通过 `pk = pairs[min(topk, n) - 1].val` 实现
  - **top-p (nucleus) 采样**: 限制累积概率不超过 p，通过 `pp = pairs[n - 1].val * topp` 实现
  - **组合采样**: 同时启用 top-k 和 top-p 时，取两者较小值 `min(pk, pp)` 作为截断边界
  - **温度缩放**: 通过 `exp(val / temperature)` 控制分布锐度，temperature > 1 使分布更均匀，< 1 使分布更尖锐

- **代码生成技术**:
  - `DESCRIPTOR(cpu)` 宏定义于 `../random_sample.h`，展开为完整的 `Descriptor` 类声明
  - 宏替换 `NAMESPACE` 为 `cpu`，生成 `op::random_sample::cpu::Descriptor` 类
  - 避免手动编写重复代码，不同设备后端（cuda/cpu/bang 等）共享相同接口
