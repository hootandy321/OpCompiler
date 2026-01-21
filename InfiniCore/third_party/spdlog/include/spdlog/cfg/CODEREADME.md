# spdlog::cfg 配置加载模块核心实现文档

spdlog::cfg 是 spdlog 日志库的配置子系统，负责从外部源（环境变量、命令行参数、字符串）解析并应用日志级别配置。该模块提供了灵活的运行时日志级别控制机制，灵感来源于 Rust 的 env_logger crate。

## 1. 模块结构

- **`helpers.h`**: 辅助函数的公共接口声明，提供 `load_levels()` API
- **`helpers-inl.h`**: 核心解析算法的实现，包含字符串处理和级别映射逻辑
- **`env.h`**: 环境变量配置加载器，从 `SPDLOG_LEVEL` 环境变量读取配置
- **`argv.h`**: 命令行参数配置加载器，从 argv 中提取 `SPDLOG_LEVEL=...` 格式的参数

## 2. 核心算法与类

### `helpers::load_levels()`
- **位置**: `helpers-inl.h` (第 73-102 行)
- **主要功能**: 从字符串解析日志级别配置并应用到 spdlog registry
- **输入格式**: `"logger1=debug,logger2=info,off"` (逗号分隔的键值对序列)
- **核心算法**:
  1. 调用 `extract_key_vals_()` 将逗号分隔的字符串解析为 `unordered_map<string, string>`
  2. 遍历键值对，对每个值调用 `level::from_str()` 进行大小写不敏感的级别解析
  3. 空键名表示全局级别，非空键名表示特定 logger 的级别
  4. 忽略无法识别的级别名称（不抛出异常）
  5. 最终调用 `details::registry::instance().set_levels()` 批量应用配置
- **时间复杂度**: O(n)，其中 n 为输入字符串中的键值对数量
- **空间复杂度**: O(n)，用于存储解析的键值对映射
- **边界处理**:
  - 空字符串或长度 >= 32768 的输入直接返回
  - 级别名称解析失败时跳过该条目
  - 大小写不敏感（通过 `to_lower_()` 转换）

### 字符串处理工具函数

#### `to_lower_(std::string &str)`
- **位置**: `helpers-inl.h` (第 23-28 行)
- **功能**: 原地转换字符串为小写
- **算法**: 使用 `std::transform` + lambda，仅转换 'A'-'Z' 范围的字符
- **时间复杂度**: O(m)，m 为字符串长度

#### `trim_(std::string &str)`
- **位置**: `helpers-inl.h` (第 31-36 行)
- **功能**: 原地去除字符串首尾空白字符（空格、换行、回车、制表符）
- **算法**: 使用 `find_last_not_of` 和 `find_first_not_of` 定位非空白边界
- **时间复杂度**: O(m)

#### `extract_kv_(char sep, const std::string &str)`
- **位置**: `helpers-inl.h` (第 45-55 行)
- **功能**: 从 `"key=val"` 格式字符串中提取键值对
- **分隔符**: 默认为 `'='`
- **返回值**: `pair<string, string>`，两个元素都经过 `trim_()` 处理
- **边界情况**:
  - 无分隔符: `("", "val")`
  - 空值: `("key", "")`
  - 带空格: `" key  =  val "` → `("key", "val")`

#### `extract_key_vals_(const std::string &str)`
- **位置**: `helpers-inl.h` (第 59-71 行)
- **功能**: 从 `"k1=v1,k2=v2,.."` 格式字符串中提取所有键值对
- **分隔符**: 外层为逗号 `,`，内层调用 `extract_kv_('=')`
- **返回值**: `unordered_map<string, string>`
- **算法**: 使用 `istringstream` + `getline` 按逗号分词
- **时间复杂度**: O(n)，n 为逗号分隔的 token 数量

### `cfg::load_env_levels()`
- **位置**: `env.h` (第 28-33 行)
- **主要功能**: 从环境变量加载日志级别配置
- **默认环境变量**: `"SPDLOG_LEVEL"`
- **实现流程**:
  1. 调用 `details::os::getenv(var)` 读取环境变量
  2. 如果环境变量非空，委托给 `helpers::load_levels()` 处理
- **使用场景**: 程序启动时根据环境变量自动配置日志级别
- **灵感来源**: Rust env_logger crate (https://crates.io/crates/env_logger)

### `cfg::load_argv_levels()`
- **位置**: `argv.h` (第 24-37 行)
- **主要功能**: 从命令行参数中提取并加载日志级别配置
- **参数格式**: `SPDLOG_LEVEL=debug` 或 `SPDLOG_LEVEL=logger1=trace,logger2=info`
- **重载版本**:
  - `load_argv_levels(int argc, const char **argv)`
  - `load_argv_levels(int argc, char **argv)` (内部转换为 const char**)
- **实现流程**:
  1. 遍历 argv[1..argc-1]
  2. 检查参数是否以 `"SPDLOG_LEVEL="` 前缀开头
  3. 提取等号后的配置字符串
  4. 委托给 `helpers::load_levels()` 处理
- **使用场景**: 允许用户在命令行动态指定日志级别

## 3. API 接口

```cpp
// 从字符串加载日志级别
namespace spdlog {
namespace cfg {
namespace helpers {
    SPDLOG_API void load_levels(const std::string &txt);
    // txt 格式示例:
    //   "debug" - 全局级别设为 debug
    //   "off,logger1=debug" - 全局关闭，仅 logger1 启用 debug
    //   "logger1=debug,logger2=info" - 多个 logger 的独立级别
}
}

// 从环境变量加载日志级别
inline void load_env_levels(const char* var = "SPDLOG_LEVEL");
// 示例: export SPDLOG_LEVEL="debug,logger1=trace"
// 调用: load_env_levels(); // 默认读取 SPDLOG_LEVEL
//       load_env_levels("MY_CUSTOM_LEVEL_VAR"); // 自定义变量名

// 从命令行参数加载日志级别
inline void load_argv_levels(int argc, const char **argv);
inline void load_argv_levels(int argc, char **argv);
// 示例: ./app "SPDLOG_LEVEL=debug,logger1=trace"
```

## 4. 使用示例

### 示例 1: 从环境变量加载
```cpp
#include <spdlog/cfg/env.h>

int main() {
    // 假设环境变量已设置: export SPDLOG_LEVEL="debug,network=trace,db=off"
    spdlog::cfg::load_env_levels(); // 自动应用配置

    // 或使用自定义环境变量
    // export MY_LOG_LEVEL="info"
    spdlog::cfg::load_env_levels("MY_LOG_LEVEL");

    // 所有 logger 的级别已根据环境变量配置
    spdlog::info("global info log");
    // ...
}
```

### 示例 2: 从命令行参数加载
```cpp
#include <spdlog/cfg/argv.h>

int main(int argc, char **argv) {
    // 假设启动命令: ./app "SPDLOG_LEVEL=debug,network=trace"
    spdlog::cfg::load_argv_levels(argc, argv);

    // 参数解析后自动配置日志级别
    spdlog::debug("debug log"); // 如果全局级别设为 debug，则输出
    // ...
}
```

### 示例 3: 编程方式直接加载
```cpp
#include <spdlog/cfg/helpers.h>

void configure_logging() {
    // 关闭所有日志，仅启用特定 logger
    spdlog::cfg::helpers::load_levels("off,network=trace,db=info");

    // 设置全局 debug 级别
    spdlog::cfg::helpers::load_levels("debug");

    // 混合配置：全局 info，特定 logger 覆盖
    spdlog::cfg::helpers::load_levels("info,http=debug,fs=warn");
}
```

### 示例 4: 完整初始化流程
```cpp
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <spdlog/cfg/argv.h>

int main(int argc, char **argv) {
    // 1. 从命令行参数加载（优先级最高）
    spdlog::cfg::load_argv_levels(argc, argv);

    // 2. 如果命令行未指定，从环境变量加载
    spdlog::cfg::load_env_levels();

    // 3. 创建 loggers
    auto network_logger = spdlog::get("network");
    auto db_logger = spdlog::get("db");

    // 4. 使用已配置的 logger
    if (network_logger) {
        network_logger->trace("network trace message");
    }
    // ...
}
```

## 5. 实现细节

### 配置语法与解析规则
- **分隔符**: 逗号 `,` 分隔多个配置项，等号 `=` 分隔 logger 名称和级别
- **全局级别**: 空键名或 `"*"` 前缀表示全局级别（如 `"debug"` 或 `"*=debug"`）
- **大小写不敏感**: 级别名称 `"DEBUG"`, `"Debug"`, `"debug"` 等价
- **空白处理**: 键和值的首尾空白字符自动去除
- **级别枚举值** (来自 `spdlog::level::level_enum`):
  - `trace` (最详细)
  - `debug`
  - `info`
  - `warn`
  - `err` (error)
  - `critical`
  - `off` (关闭所有日志)

### 内存管理
- **零拷贝优化**: 字符串处理使用 `std::string::substr` 创建临时子串
- **移动语义**: `load_levels()` 中使用 `std::move(levels)` 将解析结果传递给 registry，避免拷贝
- **容器选择**: 使用 `unordered_map` 存储键值对，提供 O(1) 查找性能

### 错误处理
- **静默失败**: 无法识别的级别名称被忽略，不抛出异常
- **输入验证**: `load_levels()` 拒绝空字符串或超长输入 (>= 32768 字节)
- **级别解析**: `level::from_str()` 返回 `level::off` 时，检查原始字符串是否为 `"off"` 以区分真实关闭和解析失败

### 线程安全
- **Registry 锁**: `details::registry::set_levels()` 内部使用互斥锁保护
- **无状态设计**: 所有配置函数都是线程安全的，不依赖可变静态状态

### 性能特性
- **一次性配置**: 配置加载通常在程序初始化阶段执行一次，非热路径
- **字符串处理开销**: O(n) 遍历和转换，使用 `std::transform` 和 `istringstream`
- **查找性能**: `unordered_map` 提供 O(1) 键值查找，`level::from_str()` 内部使用字符串比较

### 依赖关系
- **spdlog::details::registry**: 全局 logger 注册表，提供 `set_levels()` 方法
- **spdlog::details::os**: 操作系统抽象层，提供 `getenv()` 跨平台实现
- **spdlog::level**: 级别枚举定义和 `from_str()` 字符串解析函数

### Header-Only 模式支持
- **条件编译**: `helpers.h` 通过 `#ifdef SPDLOG_HEADER_ONLY` 控制是否包含 `helpers-inl.h`
- **符号导出**: `load_levels()` 使用 `SPDLOG_API` 宏，在共享库构建时正确导出符号

### 设计模式
- **Helper 模式**: `helpers` 命名空间封装核心解析逻辑，被 `env.h` 和 `argv.h` 复用
- **策略模式**: 支持多种配置源（环境变量、命令行、字符串）的统一处理流程
- **门面模式**: `load_env_levels()` 和 `load_argv_levels()` 提供简洁的高级接口，隐藏内部解析细节

### 安全边界
- **输入长度限制**: 32768 字节上限防止 DoS 攻击
- **缓冲区安全**: 使用 `std::string` 和 `std::istringstream` 避免手动内存管理
- **环境变量注入**: 依赖 `details::os::getenv()` 的安全实现
