# `spdlog::fmt::bundled` 核心实现文档

本模块是 `{fmt}` 格式化库的内置版本,作为 spdlog 日志库的依赖组件。它提供了类型安全、高性能的字符串格式化功能,采用现代 C++ 设计,支持编译时格式字符串检查和零拷贝格式化。

## 1. 模块结构

- **`core.h`**: 兼容性头文件,仅为向后兼容保留,实际引用 `format.h`
- **`args.h`**: 动态参数列表实现,支持运行时构建格式化参数存储
- **`base.h`**: 基础类型定义和核心工具(103,990 字节),包含内存管理、UTF-8 编解码、数字处理等底层基础设施
- **`chrono.h`**: 时间/日期格式化支持(79,718 字节),提供 `std::chrono` 类型和 `std::tm` 的格式化器
- **`color.h`**: 终端颜色和文本样式支持(24,290 字节),实现 ANSI 转义序列和文本样式
- **`compile.h`**: 编译时格式字符串编译(18,792 字节),通过 `constexpr` 优化实现零运行时开销
- **`format.h`**: 核心格式化功能(157,793 字节),完整的类型安全格式化系统
- **`format-inl.h`**: 内联实现细节(80,985 字节),包含 Dragonbox 浮点数算法和平台特定优化
- **`os.h`**: 操作系统抽象层(12,786 字节),文件操作和控制台输出封装
- **`ostream.h`**: `std::ostream` 集成(5,024 字节),支持流输出操作符
- **`printf.h`**: Printf 风格格式化(20,440 字节),提供传统 `printf` 语法的兼容层
- **`ranges.h`**: 范围和容器格式化(28,211 字节),支持 STL 容器和范围的格式化输出
- **`std.h`**: 标准库类型格式化(22,277 字节),为 `std::filesystem`、`std::optional`、`std::variant` 等提供格式化器
- **`xchar.h`**: 宽字符和扩展字符集支持(13,636 字节),提供 `wchar_t` 和其他字符类型的格式化

## 2. 核心组件

### `detail::dynamic_format_arg_store<Context>`
- **位置**: `args.h`
- **主要功能**: 运行时动态构建格式化参数列表,支持可变数量的参数存储和类型擦除
- **关键成员**:
  - `data_`: `std::vector<basic_format_arg<Context>>`,存储类型擦除的参数
  - `named_info_`: `std::vector<named_arg_info<char_type>>`,命名参数信息
  - `dynamic_args_`: `dynamic_arg_list`,动态分配的参数存储链表
- **核心方法**:
  - `push_back(const T& arg)`: 添加参数到存储,根据类型特征决定是否拷贝或引用
  - `push_back(std::reference_wrapper<T>)`: 添加引用参数,避免拷贝
  - `reserve(size_t, size_t)`: 预分配空间,优化性能
  - `operator basic_format_args<Context>()`: 隐式转换为格式化参数视图
- **生命周期**: RAII 模式,析构时自动清理动态分配的参数存储
- **设计模式**: 类型擦除(Type Erasure),使用 `basic_format_arg` 统一存储不同类型的参数

### `text_style`
- **位置**: `color.h`
- **主要功能**: 表示文本的显示样式,包括前景色、背景色和强调样式(粗体、斜体等)
- **关键成员**:
  - `style_`: `uint64_t`,位压缩存储样式信息(64 位紧凑编码)
  - 样式位布局:
    - `[0-23]`: 前景色值
    - `[24-25]`: 前景色类型辨别符(00=未设置, 01=RGB, 11=终端颜色)
    - `[26]`: 溢出检测位
    - `[27-52]`: 背景色(同上格式)
    - `[53]`: 背景溢出位
    - `[54-61]`: 强调样式位掩码
    - `[62-63]`: 未使用
- **核心方法**:
  - `operator|=(text_style)`: 样式组合,使用溢出位检测终端颜色冲突
  - `has_foreground()`, `has_background()`, `has_emphasis()`: 样式检测
  - `get_foreground()`, `get_background()`, `get_emphasis()`: 样式提取
- **设计模式**: 位域压缩(Bit Packing),使用溢出检测实现高效的颜色冲突验证

### `detail::duration_formatter<Char, Rep, Period>`
- **位置**: `chrono.h`
- **主要功能**: `std::chrono::duration` 的格式化实现,支持时间单位和精度控制
- **关键成员**:
  - `val`: `rep`(无符号),时间值(自动处理负数)
  - `s`: `seconds`,整秒部分
  - `precision`: `int`,小数精度(-1 表示自动)
  - `locale`: `locale_ref`,本地化设置
  - `negative`: `bool`,是否为负值
- **核心方法**:
  - `handle_nan_inf()`: 处理 NaN 和无穷大值
  - `days()`, `hour()`, `minute()`, `second()`: 时间分量提取
  - `on_duration_value()`: 格式化持续时间数值
  - `on_duration_unit()`: 格式化时间单位(如 "ms", "μs", "s", "min", "h", "d")
  - `on_24_hour()`, `on_12_hour()`, `on_minute()`, `on_second()`: 时间分量格式化
- **算法**:
  - 使用安全类型转换 `safe_duration_cast` 避免溢出
  - 支持整数和浮点表示
  - 自动选择合适的精度(微秒、毫秒、秒)

### `formatter<std::chrono::duration<Rep, Period>, Char>`
- **位置**: `chrono.h` (行 2121-2185)
- **主要功能**: `std::chrono::duration` 的格式化器特化
- **关键成员**:
  - `specs_`: `format_specs`,格式规范
  - `width_ref_`, `precision_ref_`: 动态宽度和精度参数引用
  - `fmt_`: `basic_string_view<Char>`,格式字符串
- **核心方法**:
  - `parse(parse_context&)`: 解析格式规范(宽度、精度、本地化标志、时间格式)
  - `format(duration, context)`: 格式化时间持续量
    - 创建 `duration_formatter` 实例
    - 处理动态宽度和精度
    - 使用 `parse_chrono_format` 解析时间格式字符串(如 "%H:%M:%S")
    - 应用填充、对齐和本地化

### `basic_memory_buffer<T, SIZE, Allocator>`
- **位置**: `format.h` (行 776-877)
- **主要功能**: 动态增长内存缓冲区,前 `SIZE` 个元素内联存储避免堆分配
- **关键成员**:
  - `store_`: `T[SIZE]`,内联存储数组
  - `alloc_`: `Allocator`,内存分配器
- **核心方法**:
  - `grow(buffer&, size_t)`: 静态增长函数,策略: `new_capacity = old_capacity + old_capacity / 2`
  - `resize(size_t)`: 调整大小
  - `reserve(size_t)`: 预分配容量
  - `append(ContiguousRange)`: 追加范围
- **生命周期**: RAII 模式,移动语义优化
- **性能优化**:
  - 小缓冲区优化(Small Buffer Optimization,默认 500 字节)
  - 增长因子 1.5x,平衡内存使用和重分配开销

### `detail::dragonbox` 命名空间
- **位置**: `format-inl.h`
- **主要功能**: Dragonbox 浮点数格式化算法实现,提供最优的舍入和最短输出
- **关键组件**:
  - `cache_accessor<float>` 和 `cache_accessor<double>`: 缓存 10 的幂次表
    - `pow10_significands[]`: 预计算的有效数字查找表
    - `get_cached_power(int k)`: 获取 10^k 的有效数字
  - `umul128_upper64(uint64_t, uint64_t)`: 128 位乘法的高 64 位
  - `umul192_lower128(uint64_t, uint128_fallback)`: 192 位乘法的低 128 位
  - `compute_mul(carrier_uint, cache_entry)`: 计算 `u * cache`,返回结果和是否整除标志
  - `compute_delta(cache, beta)`: 计算舍入区间边界
  - `compute_mul_parity(two_f, cache, beta)`: 计算乘法结果的奇偶性
- **算法特点**:
  - O(1) 复杂度,基于查找表和位运算
  - 生成最短舍入输出,保证往返安全(round-trip safe)
  - 支持快速路径和慢速路径切换

## 3. API 接口

```cpp
// 核心格式化函数
template <typename... T>
inline auto format(format_string<T...> fmt, T&&... args) -> std::string;
// 功能: 格式化参数并返回字符串,支持编译时格式字符串检查
// 参数: fmt - 编译时验证的格式字符串, args - 要格式化的参数
// 返回: 格式化后的字符串
// 复杂度: O(n),n 为输出字符数

template <typename OutputIt, typename... T>
auto format_to(OutputIt out, format_string<T...> fmt, T&&... args) -> OutputIt;
// 功能: 格式化参数并写入输出迭代器
// 参数: out - 输出迭代器, fmt - 格式字符串, args - 参数
// 返回: 输出结束位置的迭代器
// 支持的迭代器类型: back_insert_iterator, raw pointer, appender

template <typename... T>
void print(format_string<T...> fmt, T&&... args);
// 功能: 格式化并输出到 stdout
// 特性: 自动刷新缓冲区,支持 UTF-8 编码

template <typename... T>
void print(FILE* f, format_string<T...> fmt, T&&... args);
// 功能: 格式化并输出到文件
// 特性: 线程安全(在支持的平台),支持控制台颜色检测

// 动态参数存储
template <typename Context>
class dynamic_format_arg_store {
public:
  template <typename T> void push_back(const T& arg);
  // 功能: 动态添加参数到存储
  // 优化: reference_wrapper 避免拷贝,字符串按需拷贝

  template <typename T> void push_back(std::reference_wrapper<T> arg);
  // 功能: 添加引用参数,允许修改源数据时影响格式化结果

  template <typename T> void push_back(const named_arg<char_type, T>& arg);
  // 功能: 添加命名参数

  void clear();
  // 功能: 清空所有参数

  void reserve(size_t new_cap, size_t new_cap_named);
  // 功能: 预分配空间,减少动态分配

  operator basic_format_args<Context>() const;
  // 功能: 隐式转换为格式化参数视图
};

// 时间格式化 API
template <typename Rep, typename Period, typename Char>
struct formatter<std::chrono::duration<Rep, Period>, Char> {
  FMT_CONSTEXPR auto parse(parse_context<Char>& ctx) -> const Char*;
  // 功能: 解析时间格式规范
  // 支持的标志: L(本地化),宽度,精度,时间格式字符串(如 "%H:%M:%S.%f")

  template <typename FormatContext>
  auto format(std::chrono::duration<Rep, Period> d, FormatContext& ctx) const
      -> decltype(ctx.out());
  // 功能: 格式化时间持续量
  // 格式选项: %H(24小时), %I(12小时), %M(分钟), %S(秒), %f(微秒)
  //           %Q(数值), %q(单位), %c(日期时间), %x(本地日期), %X(本地时间)
};

// 颜色和样式 API
enum class color : uint32_t {
  alice_blue = 0xF0F8FF, antique_white = 0xFAEBD7, /* ... 158 colors */
};

enum class terminal_color : uint8_t {
  black = 30, red, green, yellow, blue, magenta, cyan, white,
  bright_black = 90, bright_red, bright_green, /* ... bright_white */
};

enum class emphasis : uint8_t {
  bold = 1, faint = 1 << 1, italic = 1 << 2, underline = 1 << 3,
  blink = 1 << 4, reverse = 1 << 5, conceal = 1 << 6, strikethrough = 1 << 7,
};

class text_style {
  FMT_CONSTEXPR text_style(emphasis em = emphasis()) noexcept;
  // 功能: 创建仅强调样式(粗体、斜体等)

  FMT_CONSTEXPR auto operator|=(text_style rhs) -> text_style&;
  // 功能: 样式组合,冲突检测(终端颜色 vs RGB 颜色)

  FMT_CONSTEXPR auto has_foreground() const noexcept -> bool;
  FMT_CONSTEXPR auto has_background() const noexcept -> bool;
  FMT_CONSTEXPR auto has_emphasis() const noexcept -> bool;
};

FMT_CONSTEXPR inline auto fg(detail::color_type foreground) noexcept -> text_style;
// 功能: 创建前景色样式
// 支持: RGB 颜色,终端颜色,枚举颜色

FMT_CONSTEXPR inline auto bg(detail::color_type background) noexcept -> text_style;
// 功能: 创建背景色样式

template <typename... T>
void print(text_style ts, format_string<T...> fmt, T&&... args);
// 功能: 使用 ANSI 转义序列打印带样式的文本
// 平台: Linux/macOS 支持,Windows 需要控制台 API
// 示例: print(fg(color::red) | emphasis::bold, "Error: {}", msg)
```

## 4. 使用示例

```cpp
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <chrono>
#include <vector>

// 示例 1: 基本格式化
void basic_formatting() {
  std::string s = fmt::format("The answer is {}.", 42);
  // 结果: "The answer is 42."

  int x = 1, y = 2;
  fmt::print("Coordinates: ({}, {})\n", x, y);
  // 输出: "Coordinates: (1, 2)"
}

// 示例 2: 动态参数存储
void dynamic_arguments() {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back(42);
  store.push_back("abc");
  store.push_back(1.5f);
  std::string result = fmt::vformat("{} and {} and {}", store);
  // 结果: "42 and abc and 1.5"
}

// 示例 3: 时间格式化
void chrono_formatting() {
  using namespace std::chrono;

  auto now = system_clock::now();
  std::string time_str = fmt::format("{:%Y-%m-%d %H:%M:%S}", now);
  // 结果: "2025-01-14 12:34:56"

  auto duration = 12345ms;
  fmt::print("Duration: {}\n", duration);
  // 输出: "Duration: 12.345s"

  fmt::print("Duration: {:%H:%M:%S}\n", duration);
  // 输出: "Duration: 00:00:12.345000"

  fmt::print("Duration: {:.2%Q %q}\n", duration);
  // 输出: "Duration: 12.35 s"
}

// 示例 4: 终端颜色和样式
void colored_output() {
  fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
             "Error: {}\n", "File not found");
  // 输出: 红色粗体 "Error: File not found"

  fmt::print(fg(fmt::color::green), "Success: {}\n", "Operation completed");
  // 输出: 绿色 "Success: Operation completed"

  auto styled_text = fmt::format(
      fg(fmt::color::steel_blue) | bg(fmt::color::navy) | fmt::emphasis::underline,
      "Styled text: {}", 42);
  // 结果: 带样式字符串 "Styled text: 42"
}

// 示例 5: 容器格式化
void range_formatting() {
  std::vector<int> v = {1, 2, 3, 4, 5};
  fmt::print("Vector: {}\n", v);
  // 输出: "Vector: [1, 2, 3, 4, 5]"

  std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
  fmt::print("Map: {}\n", m);
  // 输出: "Map: {"a": 1, "b": 2}"

  // 使用 join 自定义分隔符
  fmt::print("Joined: {}\n", fmt::join(v, " | "));
  // 输出: "Joined: 1 | 2 | 3 | 4 | 5"

  // 自定义括号和分隔符
  fmt::print("Custom: {::}\n", v);  // 无括号
  // 输出: "Custom: 1, 2, 3, 4, 5"
}

// 示例 6: 编译时格式字符串检查
void compile_time_checks() {
  // 正确: 编译通过
  fmt::print(FMT_COMPILE("{} {}"), 42, "test");

  // 错误: 编译失败 - 参数数量不匹配
  // fmt::print(FMT_COMPILE("{}"), 42, "test");

  // 错误: 编译失败 - 类型不匹配
  // fmt::print(FMT_COMPILE("{:d}"), "not an int");
}

// 示例 7: 宽字符支持
void wide_character_support() {
  std::wstring ws = fmt::format(L" wide: {}", L"测试");
  // 结果: L" wide: 测试"

  fmt::print(L"Wide: {}\n", L"字符");
  // 输出: "Wide: 字符"
}

// 示例 8: 标准库类型格式化
void std_types_formatting() {
#ifdef __cpp_lib_optional
  std::optional<int> opt = 42;
  fmt::print("Optional: {}\n", opt);
  // 输出: "Optional: optional(42)"

  std::optional<int> empty;
  fmt::print("Optional: {}\n", empty);
  // 输出: "Optional: none"
#endif

#ifdef __cpp_lib_filesystem
  std::filesystem::path p = "/tmp/test.txt";
  fmt::print("Path: {}\n", p);
  // 输出: "Path: "/tmp/test.txt""

  fmt::print("Path: {:?}\n", p);
  // 输出: 转义路径
#endif
}
```

## 5. 实现细节

### 内存管理
- **策略**: 分层内存管理,小对象内联存储,大对象动态分配
  - `basic_memory_buffer`: 内联 500 字节 + 动态扩展,增长因子 1.5x
  - `dynamic_format_arg_store`: 使用 `std::vector` 和链表组合,参数引用稳定性
  - 自定义 `allocator<T>` 使用 `malloc`/`free` 避免 C++ 运行时依赖
- **拷贝语义**:
  - 字符串: `basic_string_view` 零拷贝,`std::string` 按需拷贝
  - 参数: `reference_wrapper` 避免拷贝,其他类型按值存储
  - 移动语义全面支持,转移所有权避免拷贝

### 并发
- **线程安全**:
  - `fmt::print` 在支持的平台使用线程安全函数(`localtime_r`, `gmtime_r`)
  - `memory_buffer` 非线程安全,多线程环境需外部同步
  - `localtime`/`gmtime` 提供 RAII 包装,自动处理重入问题
- **同步原语**:
  - 不使用内部锁,依赖调用者同步
  - 文件输出使用 `flockfile`/`funlockfile`(POSIX) 或临界区(Windows)

### 性能
- **算法选择**:
  - 整数格式化: 快速除法算法,使用 `umul128` 和位运算
  - 浮点数格式化: Dragonbox + Grisu + Ryu 算法,最优舍入
  - UTF-8 编解码: Christopher Wellons 的无分支解码器
  - 时间格式化: 自定义安全类型转换,避免 `std::chrono` 隐式转换溢出
- **优化技术**:
  - 编译时格式字符串验证,零运行时检查开销
  - `constexpr` 函数支持编译时计算
  - 小对象优化(SBO),减少堆分配
  - 模板特化减少分支预测失败
  - 预计算查找表(10 的幂次表,颜色表,时间单位)
- **复杂度保证**:
  - `format`: O(n),n 为输出长度
  - `format_to`: O(n),原地写入
  - `formatted_size`: O(n),计数模式
  - Dragonbox: O(1),基于查找表

### 错误处理
- **异常策略**:
  - 使用 `FMT_THROW` 宏,支持编译时开关(`FMT_USE_EXCEPTIONS`)
  - `format_error`: 格式字符串错误
  - `system_error`: 系统 I/O 错误
  - `std::bad_alloc`: 内存分配失败
- **错误恢复**:
  - 格式化失败时输出占位符或错误消息
  - 文件写入失败时抛出异常,不静默失败
  - UTF-8 解码错误时替换为 Unicode 替换字符(U+FFFD)

### 依赖关系
- **外部依赖**:
  - 最小依赖: 仅需 `<cstring>`, `<cstdint>`, `<limits>`, `<stdexcept>`
  - 可选依赖: `<locale>`(本地化),`<filesystem>`(路径格式化),`<codecvt>`(宽字符)
  - 平台相关: `<io.h>`(Windows),`<unistd.h>`(POSIX),`<xlocale.h>`(macOS)
- **内部依赖**:
  - `base.h` 被 `format.h` 包含,提供基础工具
  - `compile.h`, `color.h`, `chrono.h`, `ranges.h`, `std.h`, `xchar.h` 等扩展 `format.h`
  - `format-inl.h` 包含实现细节,被 `format.h` 间接包含

### 设计模式
- **类型擦除(Type Erasure)**: `basic_format_arg` 统一存储不同类型,使用 `visit` 模式访问
- **策略模式(Strategy Pattern)**: `formatter` 特化,为不同类型提供定制格式化逻辑
- **工厂模式(Factory Pattern)**: `make_format_args`, `make_printf_args` 构建参数包
- **构建器模式(Builder Pattern)**: `dynamic_format_arg_store::push_back` 链式构建
- **适配器模式(Adapter Pattern)**: `basic_ostream_formatter`, `streamed_view` 适配流操作符
- **访问者模式(Visitor Pattern)**: `visit` 用于类型擦除的类型分发
- **模板元编程**:
  - 编译时格式字符串解析
  - 类型特征检测(`is_formattable`, `is_range`, `is_tuple_like`)
  - `constexpr` if 和折叠表达式实现条件编译

## 6. 关键特性总结

### 类型安全
- 编译时格式字符串验证,防止类型不匹配
- 使用 `format_string<T...>` 类型包装器,静态检查参数数量和类型
- 支持自定义类型格式化,通过特化 `formatter` 模板

### 高性能
- 零拷贝格式化(`string_view` 引用)
- Dragonbox 算法: 最优浮点数格式化
- 小缓冲区优化减少堆分配
- 编译时优化: `constexpr` 函数和模板元编程

### 可扩展性
- 用户自定义类型: 特化 `formatter<T, Char>`
- 自定义格式规范: 重载 `parse()` 和 `format()`
- 格式化参数存储: `dynamic_format_arg_store` 支持动态参数列表

### 跨平台
- 支持 Windows/Linux/macOS
- 自动检测平台特性(控制台颜色,线程安全函数)
- 提供宽字符(`wchar_t`)和多字节字符集支持
- UTF-8 编解码,支持国际化文本

### 安全性
- 格式字符串注入防护(编译时验证)
- 缓冲区溢出防护(使用 `memory_buffer` 自动管理)
- 线程安全的日志输出(在支持的平台上)
- 异常安全: RAII 和强异常安全保证
