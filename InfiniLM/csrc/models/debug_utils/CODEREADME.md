# Debug Utilities 模块核心实现文档

本模块为 InfiniLM 模型提供调试和诊断工具，用于在模型推理过程中捕获和检查中间张量值。专为开发和调试阶段设计，不应用于生产环境。

## 1. 模块结构

- **`hooks.hpp`**: 定义 Hook 回调和注册机制的核心接口，提供宏工具简化钩子管理
- **`hooks.cpp`**: 实现 HookRegistry 类的核心逻辑，包括注册、调用和模式匹配
- **`tensor_utils.hpp`**: 提供张量统计和位置记录的实用函数，支持 F32、F16、BF16 数据类型

## 2. 核心类

### `HookRegistry`
- **位置**: `hooks.hpp` (声明), `hooks.cpp` (实现)
- **主要功能**: 管理模型执行过程中的回调钩子，支持精确匹配和通配符模式匹配，允许在特定计算点捕获中间张量值
- **关键成员**:
  - `hooks_`: `std::unordered_map<std::string, HookCallback>` - 存储钩子名称到回调函数的映射表
- **核心方法**:
  - `register_hook(const std::string &name, HookCallback callback)`: 注册钩子回调，支持两种模式：精确名称（如 "layer0_q_after_proj"）和通配符模式（如 "layer0_*"）。时间复杂度 O(1)
  - `call_hook(const std::string &name, const infinicore::Tensor &tensor, int layer_idx)`: 触发钩子执行。先尝试精确匹配（O(1)），失败则遍历所有注册的模式进行前缀匹配（O(n) 其中 n 为钩子数量）。内置异常捕获机制，单个钩子失败不影响其他钩子执行
  - `clear()`: 清空所有钩子，释放 `hooks_` 映射表
  - `has_hooks()`: 检查是否有已注册的钩子，用于快速路径优化
- **生命周期**: 单例或共享指针模式，通常由模型顶层创建并向下传递至子模块。通过 `set_hook_registry()` 方法逐层传递

### `HookCallback` (函数类型)
- **位置**: `hooks.hpp`
- **主要功能**: 定义钩子回调函数签名，用于捕获和处理中间张量值
- **签名**:
  ```cpp
  using HookCallback = std::function<void(const std::string &name,
                                          const infinicore::Tensor &tensor,
                                          int layer_idx)>;
  ```
- **参数说明**:
  - `name`: 中间值的标识符（如 "layer0_q_after_proj"）
  - `tensor`: 中间张量值
  - `layer_idx`: 层索引（对于层特定钩子，不适用时为 -1）

## 3. API 接口

### 核心宏定义

```cpp
// 注册钩子回调
#define REGISTER_HOOK(registry, name, callback)
// 参数:
//   registry: HookRegistry 共享指针
//   name: 钩子名称或模式
//   callback: 回调函数对象

// 调用钩子（无层索引）
#define CALL_HOOK(registry, name, tensor)
// 自动检查 registry 是否为空和是否有钩子，避免不必要的函数调用开销

// 调用钩子（带层索引）
#define CALL_HOOK_LAYER(registry, name, tensor, layer_idx)
// 用于需要指定层索引的场景（如 Transformer 层）
```

### 模型集成宏

```cpp
// 声明钩子成员变量
#define HOOK_REGISTRY_MEMBER()
// 在类中添加: hook_registry_ 和 hook_prefix_ 成员

// 简单设置钩子注册表（不转发到子模块）
#define SET_HOOK_REGISTRY_SIMPLE()
// 实现 set_hook_registry() 方法

// 设置钩子注册表并转发到 1-2 个子模块
#define SET_HOOK_REGISTRY(...)
// 可变参数宏，支持 1 或 2 个子模块
// 自动为每个子模块构建递增前缀（如 "layer0" -> "layer0_attention"）

// 设置钩子注册表并转发到子模块向量
#define SET_HOOK_REGISTRY_VEC(vec_name)
// 为向量中的每个元素添加索引前缀（如 "layer0", "layer1", ...）
// 如果父级有前缀，则变为 "parent_layer0", "parent_layer1", ...
```

### 张量日志函数

```cpp
// 记录张量统计信息和样本值
inline void log_tensor_stats(const infinicore::Tensor &tensor,
                             const std::string &name,
                             bool log_samples = true,
                             size_t max_samples = 10);
// 功能:
//   - 记录张量形状、数据类型、设备信息
//   - 对于浮点类型（F32/F16/BF16），计算 min/max/mean/numel
//   - 记录前 N 个和后 N 个样本值（BF16 专用，用于调试解码步骤）
//   - 自动将张量复制到 CPU 进行计算

// 记录张量特定位置的值
inline void log_tensor_positions(const infinicore::Tensor &tensor,
                                 const std::string &name,
                                 const std::vector<std::vector<size_t>> &positions);
// 功能:
//   - 计算多维索引的线性偏移量
//   - 验证位置有效性
//   - 记录指定位置的标量值
```

## 4. 使用示例

### 示例 1: 基本钩子注册和调用

```cpp
#include "debug_utils/hooks.hpp"
#include "debug_utils/tensor_utils.hpp"
#include <spdlog/spdlog.h>

using namespace infinilm::models::debug_utils;

// 创建钩子注册表
auto hook_registry = std::make_shared<HookRegistry>();

// 注册精确匹配钩子
hook_registry->register_hook("layer0_q_after_proj", [](const std::string &name,
                                                       const infinicore::Tensor &tensor,
                                                       int layer_idx) {
    log_tensor_stats(tensor, name);
    SPDLOG_INFO("Layer {} hook triggered: {}", layer_idx, name);
});

// 注册通配符模式钩子（捕获所有 layer0_ 开头的钩子）
hook_registry->register_hook("layer0_*", [](const std::string &name,
                                            const infinicore::Tensor &tensor,
                                            int layer_idx) {
    SPDLOG_INFO("Pattern match caught: {} at layer {}", name, layer_idx);
});

// 在模型代码中调用钩子
infinicore::Tensor intermediate_tensor = /* ... */;
CALL_HOOK_LAYER(hook_registry, "layer0_q_after_proj", intermediate_tensor, 0);

// 清理
hook_registry->clear();
```

### 示例 2: 在 Transformer 层中集成钩子

```cpp
class TransformerLayer {
public:
    // ... 其他成员 ...

    // 使用宏声明钩子成员
    HOOK_REGISTRY_MEMBER()

    // 使用宏设置钩子注册表并转发到子模块
    SET_HOOK_REGISTRY(attention, mlp)  // 自动为 attention 和 mlp 设置前缀

    // 或使用向量版本
    // SET_HOOK_REGISTRY_VEC(layers)

    void forward(const infinicore::Tensor &x) {
        // 计算 QKV 投影
        auto q = q_proj(x);
        // 触发钩子
        CALL_HOOK_LAYER(hook_registry_, "q_after_proj", q, layer_idx_);

        auto k = k_proj(x);
        CALL_HOOK_LAYER(hook_registry_, "k_after_proj", k, layer_idx_);

        // ... 继续计算 ...
    }

private:
    int layer_idx_;
    std::shared_ptr<Attention> attention_;
    std::shared_ptr<MLP> mlp_;
};

// 在模型顶层使用
auto model = std::make_shared<Transformer>();
model->set_hook_registry(hook_registry, "model");  // 设置前缀
```

### 示例 3: 使用张量日志调试

```cpp
#include "debug_utils/tensor_utils.hpp"

void debug_attention_output(const infinicore::Tensor &attn_output) {
    // 记录完整统计信息和样本
    log_tensor_stats(attn_output, "attention_output",
                    true,  // 记录样本值
                    10);   // 最多 10 个样本

    // 记录特定位置（如最后一个 token 的注意力分数）
    std::vector<std::vector<size_t>> positions = {
        {0, 31, 0},   // [batch, seq_len, head_dim]
        {0, 31, 63},  // 最后一个 token 的最后两个维度
    };
    log_tensor_positions(attn_output, "attention_output", positions);
}
```

### 示例 4: 调试 BF16 解码精度问题

```cpp
void debug_decode_step(const infinicore::Tensor &logits) {
    // BF16 类型会自动记录最后 N 个值，用于检查新解码的 token
    log_tensor_stats(logits, "decode_logits",
                    true,  // 记录样本
                    5);    // 前 5 个和后 5 个
    // 输出会包含:
    //   - "Sample values (first 5):"
    //   - "Sample values (last 5):" <- 关键用于查看新生成的 token logits
}
```

## 5. 实现细节

### 内存管理
- **共享所有权**: HookRegistry 使用 `std::shared_ptr` 管理，允许多个子模块共享同一注册表
- **引用捕获**: HookCallback 通过 `std::function` 存储，支持捕获外部状态（但需注意生命周期）
- **自动设备转换**: `log_tensor_stats()` 自动将张量复制到 CPU（使用 `tensor->to(Device::CPU)`），避免跨设备内存访问

### 并发性
- **非线程安全**: HookRegistry 的 `hooks_` 映射表未使用锁保护。假设钩子注册仅在模型初始化阶段单线程进行
- **调用阶段安全性**: 钩子调用时仅读取映射表，多线程并发调用不同钩子是安全的（假设底层 Tensor 操作是线程安全的）
- **异常隔离**: 单个钩子的异常不会影响其他钩子执行，但异常会通过 `SPDLOG_ERROR` 记录

### 性能考虑
- **快速路径优化**: `CALL_HOOK` 宏在调用前检查 `has_hooks()`，避免无钩子时的函数调用开销
- **模式匹配复杂度**: 通配符匹配是 O(n) 线性搜索。建议在调试后移除钩子或使用精确匹配
- **数据类型优化**: F16 和 BF16 转换为 F32 时分配临时 `std::vector<float>`，对大张量可能有内存开销
- **最小日志开销**: 使用 `SPDLOG_DEBUG` 和 `SPDLOG_INFO`，可在生产构建中完全禁用

### 错误处理
- **异常捕获**: 钩子调用通过 `try-catch` 包装，防止用户回调崩溃整个系统
- **日志记录**: 所有错误通过 spdlog 记录，包括钩子名称和异常信息
- **边界检查**: `log_tensor_positions()` 验证索引有效性，防止越界访问

### 依赖关系
- **外部依赖**:
  - `infinicore/Tensor.hpp`: 核心张量抽象
  - `spdlog/spdlog.h`: 日志框架
- **内部依赖**: 无（独立工具模块）
- **未来计划**: 注释提到可能将 HookRegistry 移至 InfiniCore 作为通用工具

### 设计模式
- **观察者模式**: HookRegistry 实现观察者模式，模型是主题，钩子是观察者
- **策略模式**: HookCallback 允许运行时插入不同的调试策略
- **宏元编程**: 大量使用宏减少样板代码（`SET_HOOK_REGISTRY` 系列宏），利用宏参数计数和 `##` 令牌拼接实现可变参数
- **前缀构建模式**: 通过 `BUILD_HOOK_PREFIX` 递增构建层次化钩子名称（如 "model_layer0_attention_q"）

### 算法细节
- **F16 到 F32 转换**: 使用位级操作实现快速转换
  ```cpp
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  ```
  利用 F16 和 F32 的指数偏移差值（F16 偏移 15，F32 偏移 127，差 112）

- **BF16 到 F32 转换**: 简单的左移 16 位
  ```cpp
  uint32_t f32 = (static_cast<uint32_t>(b) << 16);
  ```
  利用 BF16 和 F32 共享相同指数偏移（127）的特性

- **多维索引到线性索引**: 使用行优先布局
  ```cpp
  size_t idx = 0;
  size_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
      idx += pos[i] * stride;
      stride *= shape[i];
  }
  ```

### 宏技术细节
- **可变参数宏**: 使用 `__VA_ARGS__` 和多层宏展开实现参数计数
  ```cpp
  #define SET_HOOK_REGISTRY(...) \
      SET_HOOK_REGISTRY_IMPL(__VA_ARGS__)

  #define SET_HOOK_REGISTRY_IMPL(...) \
      SET_HOOK_REGISTRY_GET_NTH(__VA_ARGS__, SET_HOOK_REGISTRY_2, SET_HOOK_REGISTRY_1, SET_HOOK_REGISTRY_0,)(__VA_ARGS__)
  ```
  通过传递 "选择器" 参数（SET_HOOK_REGISTRY_2/1/0）并根据参数数量选择正确的实现

- **令牌拼接**: 使用 `##` 操作符动态生成成员变量名
  ```cpp
  submodule##_->set_hook_registry(...)  // 如果 submodule 是 "attention"，生成 attention_->set_hook_registry(...)
  ```
