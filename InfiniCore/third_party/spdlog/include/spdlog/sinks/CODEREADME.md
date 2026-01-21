# spdlog Sinks Module Core Implementation Documentation

spdlog sinks模块是spdlog日志库的核心输出目标组件系统,提供了30多种不同的日志输出目标实现。该模块采用策略模式设计,允许日志消息被路由到各种输出目标(文件、控制台、网络、数据库等),每个sink负责将格式化后的日志消息写入特定的目标。模块基于CRTP和模板设计实现线程安全和性能优化。

## 1. Module Structure

该目录包含34个头文件,实现了spdlog的所有日志输出目标(sink):

**核心基础文件**:
- `sink.h` / `sink-inl.h`: 基础sink抽象接口,定义所有sink的公共API
- `base_sink.h` / `base_sink-inl.h`: CRTP基类模板,提供锁管理和格式化功能

**标准输出sinks**:
- `stdout_sinks.h` / `stdout_sinks-inl.h`: 标准输出/错误输出sink(无颜色)
- `stdout_color_sinks.h` / `stdout_color_sinks-inl.h`: 带颜色的控制台输出sink(跨平台抽象)
- `ansicolor_sink.h` / `ansicolor_sink-inl.h`: ANSI转义序列彩色终端sink(Unix/Linux)
- `wincolor_sink.h` / `wincolor_sink-inl.h`: Windows控制台彩色sink
- `msvc_sink.h`: Microsoft Visual Studio调试输出sink
- `win_eventlog_sink.h`: Windows事件日志sink

**文件sinks**:
- `basic_file_sink.h` / `basic_file_sink-inl.h`: 基础单文件sink
- `rotating_file_sink.h` / `rotating_file_sink-inl.h`: 基于大小的文件轮转sink
- `daily_file_sink.h`: 基于日期的文件轮转sink(每日轮转)
- `hourly_file_sink.h`: 基于小时的文件轮转sink

**网络sinks**:
- `tcp_sink.h`: TCP客户端网络sink
- `udp_sink.h`: UDP客户端网络sink
- `syslog_sink.h`: Unix syslog sink
- `systemd_sink.h`: systemd journal sink

**特殊目标sinks**:
- `null_sink.h`: 空sink(丢弃所有日志,用于测试)
- `ostream_sink.h`: C++ ostream输出sink
- `callback_sink.h`: 自定义回调函数sink
- `ringbuffer_sink.h`: 环形缓冲区内存sink
- `dist_sink.h`: 分发sink(将日志分发到多个子sink)
- `dup_filter_sink.h`: 重复消息过滤sink

**第三方集成sinks**:
- `android_sink.h`: Android logcat sink
- `kafka_sink.h`: Apache Kafka消息队列sink
- `mongo_sink.h`: MongoDB数据库sink
- `qt_sinks.h`: Qt框架GUI sink(QTextEdit/QPlainTextEdit)

## 2. Core Classes

### `sink` (Abstract Base)
- **Location**: `sink.h`
- **Primary Function**: 定义所有sink的抽象接口,是整个sink继承体系的根节点
- **Key Members**:
  - `level_`: `level_t`类型的原子变量,存储sink的日志级别阈值,使用`std::atomic<level::level_enum>`实现无锁并发访问
- **Core Methods**:
  - `log(const details::log_msg &msg)`: 纯虚函数,子类必须实现,将日志消息写入目标
  - `flush()`: 纯虚函数,刷新缓冲区到目标设备
  - `set_pattern(const std::string &pattern)`: 设置日志格式模式字符串
  - `set_formatter(std::unique_ptr<spdlog::formatter>)`: 设置自定义格式化器
  - `set_level(level::level_enum)`: 设置日志级别阈值,使用`memory_order_relaxed`内存序
  - `should_log(level::level_enum)`: 检查消息级别是否达到阈值,使用`memory_order_relaxed`优化性能
  - `level()`: 获取当前日志级别阈值

**Design Pattern**: 接口隔离原则,只定义必要的方法签名。级别检查使用`std::atomic::load`和`std::atomic::store`配合`memory_order_relaxed`,保证原子性同时避免内存屏障开销。

### `base_sink<Mutex>` (CRTP Template Base)
- **Location**: `base_sink.h` / `base_sink-inl.h`
- **Primary Function**: 使用CRTP(Curiously Recurring Template Pattern)实现带锁管理的基类,处理线程安全和格式化逻辑,派生类只需实现`sink_it_()`和`flush_()`
- **Key Members**:
  - `formatter_`: `std::unique_ptr<spdlog::formatter>`,存储日志格式化器,默认为`pattern_formatter`
  - `mutex_`: `Mutex`类型模板参数,通常是`std::mutex`或`details::null_mutex`,实现可选的线程安全
- **Core Methods**:
  - `log(const details::log_msg &msg)`: final方法,使用`std::lock_guard<Mutex>`锁定mutex_后调用虚函数`sink_it_()`
  - `flush()`: final方法,加锁后调用虚函数`flush_()`
  - `set_pattern(const std::string &pattern)`: final方法,加锁后调用虚函数`set_pattern_()`
  - `set_formatter(std::unique_ptr<formatter>)`: final方法,加锁后调用虚函数`set_formatter_()`
  - `sink_it_(const details::log_msg &msg)`: 纯虚函数,派生类实现实际写入逻辑
  - `flush_()`: 纯虚函数,派生类实现实际刷新逻辑
  - `set_pattern_(const std::string &)`: 虚函数,默认创建pattern_formatter实例
  - `set_formatter_(std::unique_ptr<formatter>)`: 虚函数,转移ownership到formatter_

**Lifecycle**: 构造函数默认创建`pattern_formatter`,析构函数为default。拷贝构造和移动构造均被删除,防止意外的共享状态。

**Design Pattern**: CRTP + Template Method模式。`Mutex`模板参数允许编译时选择线程安全策略:多线程环境使用`std::mutex`,单线程环境使用`null_mutex`(零开销)。所有public方法都通过`std::lock_guard`保护临界区。

### `stdout_sink_base<ConsoleMutex>`
- **Location**: `stdout_sinks.h` / `stdout_sinks-inl.h`
- **Primary Function**: 控制台输出的基础实现,支持stdout和stderr,跨平台处理Windows和Unix的差异
- **Key Members**:
  - `mutex_`: `typename ConsoleMutex::mutex_t&`引用,绑定到全局控制台互斥锁(`console_mutex`或`console_nullmutex`)
  - `file_`: `FILE*`指针,指向标准输出流(stdout或stderr)
  - `formatter_`: 格式化器对象
  - `handle_` (Windows only): `HANDLE`类型,存储Windows文件句柄,用于`WriteFile`API
- **Core Methods**:
  - `stdout_sink_base(FILE *file)`: 构造函数,初始化mutex_引用、file_、formatter_,Windows下通过`_get_osfhandle`获取HANDLE
  - `log(const details::log_msg &msg)`: Windows下使用`WriteFile`API避免`\r\r\n`问题(issue #1675),Unix下使用`fwrite`;每次写入后调用`fflush`确保终端立即显示
  - `flush()`: 调用`fflush(file_)`
  - `set_pattern/set_formatter`: 加锁后替换formatter_

**Platform Specifics**: Windows实现检测`INVALID_HANDLE_VALUE`处理无控制台情况;Unix实现直接使用`fwrite`。每次log后强制flush以支持实时监控。

### `ansicolor_sink<ConsoleMutex>`
- **Location**: `ansicolor_sink.h` / `ansicolor_sink-inl.h`
- **Primary Function**: 在Unix/Linux终端使用ANSI转义序列实现彩色日志输出,自动检测终端能力
- **Key Members**:
  - `target_file_`: `FILE*`,输出目标文件(stdout或stderr)
  - `colors_`: `std::array<std::string, level::n_levels>`,存储每个日志级别的ANSI颜色代码
  - `should_do_colors_`: `bool`,标记是否启用颜色输出
- **ANSI Color Constants**: 定义了完整的ANSI转义序列常量集:
  - 格式代码: `reset`, `bold`, `dark`, `underline`, `blink`, `reverse`, `concealed`, `clear_line`
  - 前景色: `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`
  - 背景色: `on_black`到`on_white`
  - 组合样式: `yellow_bold`, `red_bold`, `bold_on_red`
- **Core Methods**:
  - `set_color_mode(color_mode mode)`: 设置颜色模式(automatic/always/never),automatic模式下调用`details::os::in_terminal()`和`details::os::is_color_terminal()`检测能力
  - `log(const details::log_msg &msg)`: 格式化消息后,根据`msg.color_range_start/end`定位颜色区间,使用`print_ccode_()`插入ANSI代码,最后输出`reset`恢复默认颜色
  - `set_color(level::level_enum, string_view_t color)`: 自定义特定级别的颜色

**Default Colors**: trace=white, debug=cyan, info=green, warn=yellow_bold, err=red_bold, critical=bold_on_red

**Performance**: 颜色检测在构造时完成一次,运行时只需检查`should_do_colors_`标志。

### `wincolor_sink<ConsoleMutex>`
- **Location**: `wincolor_sink.h` / `wincolor_sink-inl.h`
- **Primary Function**: Windows平台使用`SetConsoleTextAttribute` API实现彩色控制台输出
- **Key Members**:
  - `out_handle_`: `void*`,存储Windows控制台句柄(`HANDLE`)
  - `colors_`: `std::array<std::uint16_t, level::n_levels>`,存储每个级别的Windows控制台属性(FOREGROUND_xxx组合)
- **Core Methods**:
  - `set_color_mode_impl(color_mode mode)`: automatic模式下使用`GetConsoleMode`检测是否为真实控制台
  - `log(const details::log_msg &msg)`: 调用`set_foreground_color_()`获取原始属性→调用`SetConsoleTextAttribute`设置新颜色→输出着色文本→恢复原始属性
  - `set_foreground_color_(std::uint16_t attribs)`: 使用`GetConsoleScreenBufferInfo`获取当前属性,修改低4位(foreground bits),返回原始属性用于恢复
  - `print_range_(const memory_buf_t &formatted, size_t start, size_t end)`: Windows下使用`WriteConsoleA`或`WriteConsoleW`(UTF-8模式)输出

**Platform Specifics**: Windows控制台属性使用位掩码:FOREGROUND_RED/GREEN/BLUE(0x1/0x2/0x4),FOREGROUND_INTENSITY(0x8)。

**UTF-8 Support**: `SPDLOG_UTF8_TO_WCHAR_CONSOLE`宏启用时,先将UTF-8转换为UTF-16再使用`WriteConsoleW`输出。

### `basic_file_sink<Mutex>`
- **Location**: `basic_file_sink.h` / `basic_file_sink-inl.h`
- **Primary Function**: 最简单的文件日志sink,将所有日志写入单个文件
- **Key Members**:
  - `file_helper_`: `details::file_helper`对象,封装文件操作(打开/写入/刷新/关闭)
- **Core Methods**:
  - `basic_file_sink(filename_t filename, bool truncate, const file_event_handlers &)`: 构造函数,调用`file_helper_.open()`打开文件,truncate=true时清空文件
  - `sink_it_(const details::log_msg &msg)`: 使用formatter_格式化消息到`memory_buf_t`,调用`file_helper_.write()`写入
  - `flush_()`: 调用`file_helper_.flush()`
  - `truncate()`: 调用`file_helper_.reopen(true)`重新打开并清空文件

**File Events**: 通过`file_event_handlers`支持文件打开/关闭前的回调,可用于自定义预处理。

### `rotating_file_sink<Mutex>`
- **Location**: `rotating_file_sink.h` / `rotating_file_sink-inl.h`
- **Primary Function**: 基于文件大小的自动轮转sink,当文件超过`max_size`时触发轮转,保留最多`max_files`个历史文件
- **Key Members**:
  - `base_filename_`: 基础文件名(如"logs/mylog.txt")
  - `max_size_`: `std::size_t`,单个文件最大字节数
  - `max_files_`: `std::size_t`,保留的历史文件数量上限,最大200000(`MaxFiles`常量)
  - `current_size_`: `std::size_t`,当前文件的已写字节数,在构造时通过`file_helper_.size()`获取(昂贵操作,只执行一次)
  - `file_helper_`: 文件操作辅助对象
- **Core Methods**:
  - `rotating_file_sink(filename_t, std::size_t max_size, std::size_t max_files, bool rotate_on_open, ...)`: 构造函数,验证max_size>0和max_files<=MaxFiles,打开文件,rotate_on_open=true时立即轮转
  - `sink_it_(const details::log_msg &msg)`: 计算new_size=current_size+formatted.size(),若new_size>max_size_则flush并检查真实文件大小>0(处理磁盘满issue #2261),若满足则调用`rotate_()`,写入消息并更新current_size_
  - `rotate_()`: 实现文件轮转算法:
    1. 调用`file_helper_.close()`关闭当前文件
    2. 循环从max_files_倒序到1: 对每个i,调用`calc_filename(base_filename_, i-1)`计算源文件名,若存在则重命名为`calc_filename(base_filename_, i)`
    3. Windows重命名失败时等待100ms后重试(workaround antivirus问题),若仍失败则truncate当前文件并抛出异常
    4. 调用`file_helper_.reopen(true)`重新打开并清空当前文件,重置current_size_=0
  - `calc_filename(const filename_t &filename, std::size_t index)`: 静态方法,将"mylog.txt"转换为"mylog.1.txt"、"mylog.2.txt"等,使用`details::file_helper::split_by_extension()`分离扩展名
  - `rename_file_(src, target)`: 删除target(如果存在)→重命名src→target,返回成功标志

**Rotation Algorithm**: 文件命名遵循递增序列: log.txt → log.1.txt → log.2.txt → ... → log.N.txt (删除)

**Performance**: current_size_是估算值,只在轮转时同步真实文件大小,避免频繁调用`stat()`系统调用。

### `daily_file_sink<Mutex, FileNameCalc>`
- **Location**: `daily_file_sink.h`
- **Primary Function**: 基于时间的每日轮转sink,在指定时间(默认午夜0:0)触发轮转,支持最多`max_files`个历史文件
- **Key Members**:
  - `base_filename_`: 基础文件名模板
  - `rotation_h_` / `rotation_m_`: 轮转时间点(小时/分钟)
  - `rotation_tp_`: `log_clock::time_point`,下一次轮转的精确时间点
  - `filenames_q_`: `details::circular_q<filename_t>`,循环队列存储历史文件名,用于max_files限制
  - `truncate_`: `bool`,轮转时是否清空文件
- **Core Methods**:
  - `daily_file_sink(filename_t, int rotation_hour, int rotation_minute, bool truncate, uint16_t max_files, ...)`: 构造函数,验证时间合法性(0-23小时,0-59分钟),调用`FileNameCalc::calc_filename()`计算初始文件名,调用`next_rotation_tp_()`计算下次轮转时间,若max_files>0则调用`init_filenames_q_()`扫描现有文件
  - `sink_it_(const details::log_msg &msg)`: 检查`msg.time >= rotation_tp_`,若满足则计算新文件名→打开文件→更新rotation_tp_,格式化并写入消息,若发生轮转且max_files>0则调用`delete_old_()`清理最旧文件
  - `next_rotation_tp_()`: 获取当前时间→设置tm_hour/minute为rotation_h_/rotation_m_,tm_sec=0→转换为time_point,若时间已过则加24小时
  - `delete_old_()`: 若`filenames_q_.full()`,取出front()文件名→调用`remove_if_exists()`删除→若失败则将当前文件push_back并抛出异常,否则将当前文件push_back到队列

**Filename Calculators**: 两种策略:
- `daily_filename_calculator`: 生成"basename.YYYY-MM-DD.ext"格式(如"mylog.2025-01-14.log")
- `daily_filename_format_calculator`: 使用strftime格式字符串,支持自定义格式如"myapp-%Y-%m-%d:%H:%M:%S.log"

**Initialization**: `init_filenames_q_()`回溯扫描最多max_files个历史文件,按时间倒序填充循环队列。

### `dist_sink<Mutex>`
- **Location**: `dist_sink.h`
- **Primary Function**: 分发器sink,将每条日志消息复制并转发到多个子sink,实现多目标日志记录
- **Key Members**:
  - `sinks_`: `std::vector<std::shared_ptr<sink>>`,存储所有子sink的智能指针
- **Core Methods**:
  - `add_sink(std::shared_ptr<sink> sub_sink)`: 加锁后push_back到sinks_
  - `remove_sink(std::shared_ptr<sink> sub_sink)`: 使用`std::remove`配合`erase`移除指定sink
  - `set_sinks(std::vector<std::shared_ptr<sink>> sinks)`: 替换整个sink列表
  - `sink_it_(const details::log_msg &msg)`: 遍历sinks_,对每个sub_sink调用`should_log(msg.level)`检查级别,若通过则调用`sub_sink->log(msg)`
  - `flush_()`: 遍历sinks_,调用每个`sub_sink->flush()`
  - `set_formatter_(std::unique_ptr<formatter>)`: 先更新自身formatter_,然后遍历sinks_,为每个sub_sink调用`set_formatter(formatter_->clone())`克隆格式化器

**Use Cases**: 同时输出到文件和控制台、实现多级日志备份、与`dup_filter_sink`组合实现重复过滤。

### `dup_filter_sink<Mutex>`
- **Location**: `dup_filter_sink.h`
- **Primary Function**: 重复消息过滤器,在指定时间内若连续收到相同消息则合并为一条"Skipped N duplicate messages"日志
- **Key Members**:
  - `max_skip_duration_`: `std::chrono::microseconds`,允许的最大重复时间窗口
  - `last_msg_time_`: `log_clock::time_point`,上一条消息的时间戳
  - `last_msg_payload_`: `std::string`,上一条消息的payload内容
  - `skip_counter_`: `size_t`,当前跳过的重复消息计数
  - `skipped_msg_log_level_`: `level::level_enum`,被跳过消息的日志级别
- **Core Methods**:
  - `dup_filter_sink(std::chrono::duration<Rep, Period> max_skip_duration)`: 构造函数,初始化max_skip_duration_
  - `sink_it_(const details::log_msg &msg)`: 调用`filter_(msg)`判断是否过滤:
    - 若返回false(重复): skip_counter_++,保存skipped_msg_log_level_,直接return
    - 若返回true(非重复): 若skip_counter_>0,构造"Skipped %u duplicate messages.."消息调用`dist_sink::sink_it_()`输出汇总日志,然后输出当前消息,更新last_msg_time_/last_msg_payload_,重置skip_counter_=0
  - `filter_(const details::log_msg &msg)`: 计算`filter_duration = msg.time - last_msg_time_`,返回`(filter_duration > max_skip_duration_) || (msg.payload != last_msg_payload_)`

**Application Scenario**: 防止高频重复日志(如连接失败重试)淹没控制台,减少日志量同时保留关键信息。

### `ringbuffer_sink<Mutex>`
- **Location**: `ringbuffer_sink.h`
- **Primary Function**: 内存环形缓冲区sink,保留最近的N条日志用于调试和崩溃现场分析
- **Key Members**:
  - `q_`: `details::circular_q<details::log_msg_buffer>`,循环队列存储日志消息
- **Core Methods**:
  - `ringbuffer_sink(size_t n_items)`: 构造函数,初始化容量为n_items的循环队列,验证n_items>0
  - `sink_it_(const details::log_msg &msg)`: 构造`log_msg_buffer{msg}`并push_back到q_,自动覆盖最旧元素
  - `last_raw(size_t lim = 0)`: 返回最近lim条原始日志消息(未格式化),lim=0时返回全部
  - `last_formatted(size_t lim = 0)`: 返回最近lim条格式化后的字符串,遍历队列调用formatter_->format()

**Use Cases**: 程序崩溃时dump最近日志、GUI实时显示最近日志流、内存中的滚动日志窗口。

### `tcp_sink<Mutex>`
- **Location**: `tcp_sink.h`
- **Primary Function**: TCP客户端sink,连接到远程服务器并发送格式化后的日志,支持自动重连
- **Key Members**:
  - `config_`: `tcp_sink_config`,存储服务器地址、端口、超时时间、lazy_connect标志
  - `client_`: `details::tcp_client`,封装TCP socket操作的平台相关实现
- **Core Methods**:
  - `tcp_sink(std::string host, int port, int timeout_ms, bool lazy_connect)`: 构造函数,若非lazy_connect则立即调用`client_.connect(host, port, timeout_ms)`
  - `sink_it_(const details::log_msg &msg)`: 格式化消息→检查`client_.is_connected()`→若未连接则调用`connect()`→调用`client_.send(data, size)`发送

**Reconnection**: 每次log时检查连接状态,断线自动重连(阻塞式,超时由timeout_ms控制)。

**Platform Specifics**: 使用`#ifdef _WIN32`选择`tcp_client-windows.h`或`tcp_client.h`实现。

### `syslog_sink<Mutex>`
- **Location**: `syslog_sink.h`
- **Primary Function**: Unix syslog sink,通过标准C库`syslog()`函数发送日志到系统日志服务
- **Key Members**:
  - `syslog_levels_`: `std::array<int, 7>`,映射spdlog级别到syslog优先级(LOG_DEBUG/LOG_INFO/LOG_WARNING/LOG_ERR/LOG_CRIT)
  - `ident_`: `std::string`,syslog标识符(程序名)
  - `enable_formatting_`: `bool`,是否使用spdlog格式化器或直接输出raw payload
- **Core Methods**:
  - `syslog_sink(std::string ident, int syslog_option, int syslog_facility, bool enable_formatting)`: 构造函数,初始化级别映射数组,调用`::openlog(ident, syslog_option, syslog_facility)`打开syslog连接
  - `~syslog_sink()`: 调用`::closelog()`关闭连接
  - `sink_it_(const details::log_msg &msg)`: 根据enable_formatting_选择格式化或直接使用payload,调用`syslog_prio_from_level(msg)`获取优先级,调用`::syslog(prio, "%.*s", length, payload.data)`发送
  - `syslog_prio_from_level(const details::log_msg &msg)`: 虚函数,可被子类重写自定义映射逻辑

**syslog() Mapping**: trace/debug→LOG_DEBUG, info→LOG_INFO, warn→LOG_WARNING, err→LOG_ERR, critical→LOG_CRIT

## 3. API Interface

```cpp
// ========== 基础Sink接口 ==========

class sink {
public:
    virtual ~sink() = default;

    // 写入日志消息(纯虚函数,子类必须实现)
    virtual void log(const details::log_msg &msg) = 0;

    // 刷新缓冲区(纯虚函数,子类必须实现)
    virtual void flush() = 0;

    // 设置格式模式字符串(纯虚函数)
    virtual void set_pattern(const std::string &pattern) = 0;

    // 设置自定义格式化器(纯虚函数)
    virtual void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) = 0;

    // 设置日志级别阈值(线程安全,使用memory_order_relaxed)
    void set_level(level::level_enum log_level);

    // 获取日志级别阈值
    level::level_enum level() const;

    // 检查消息级别是否达到阈值(高效,无锁)
    bool should_log(level::level_enum msg_level) const;

protected:
    level_t level_{level::trace};  // 原子变量存储级别阈值
};

// ========== CRTP基类模板 ==========

template <typename Mutex>
class base_sink : public sink {
public:
    base_sink();  // 默认构造,创建pattern_formatter
    explicit base_sink(std::unique_ptr<spdlog::formatter> formatter);  // 自定义formatter

    // Final方法,加锁后调用虚函数实现
    void log(const details::log_msg &msg) final;
    void flush() final;
    void set_pattern(const std::string &pattern) final;
    void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) final;

protected:
    // 子类实现实际写入逻辑(纯虚函数)
    virtual void sink_it_(const details::log_msg &msg) = 0;
    virtual void flush_() = 0;

    // 可选重写的虚函数
    virtual void set_pattern_(const std::string &pattern);
    virtual void set_formatter_(std::unique_ptr<spdlog::formatter> sink_formatter);

    std::unique_ptr<spdlog::formatter> formatter_;  // 格式化器
    Mutex mutex_;  // 互斥锁(可能是std::mutex或null_mutex)
};

// ========== 工厂函数示例 ==========

// 创建每日轮转文件logger
template <typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<logger> daily_logger_mt(
    const std::string &logger_name,
    const filename_t &filename,
    int hour = 0,
    int minute = 0,
    bool truncate = false,
    uint16_t max_files = 0,
    const file_event_handlers &event_handlers = {});

// 创建大小轮转文件logger
template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> rotating_logger_mt(
    const std::string &logger_name,
    const filename_t &filename,
    size_t max_file_size,
    size_t max_files,
    bool rotate_on_open = false,
    const file_event_handlers &event_handlers = {});

// 创建带颜色的标准输出logger
template <typename Factory = spdlog::synchronous_factory>
std::shared_ptr<logger> stdout_color_mt(
    const std::string &logger_name,
    color_mode mode = color_mode::automatic);

// ========== 类型别名惯例 ==========

// 所有sink都提供_mt(多线程)和_st(单线程)版本
using basic_file_sink_mt = basic_file_sink<std::mutex>;
using basic_file_sink_st = basic_file_sink<details::null_mutex>;
using rotating_file_sink_mt = rotating_file_sink<std::mutex>;
using rotating_file_sink_st = rotating_file_sink<details::null_mutex>;
```

## 4. Usage Example

```cpp
#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/callback_sink.h>

// 示例1: 创建基础每日轮转文件logger
void example_daily_file_logger() {
    // 创建每日午夜轮转的文件sink,保留最近7天的日志
    auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
        "logs/daily.log", 0, 0, false, 7);

    // 创建logger并设置级别
    auto logger = std::make_shared<spdlog::logger>("daily_logger", daily_sink);
    logger->set_level(spdlog::level::debug);

    // 使用工厂函数简化创建
    auto logger2 = spdlog::daily_logger_mt("daily_logger2", "logs/app.log", 0, 0);

    // 记录日志
    logger2->info("Application started");
    logger2->debug("Debugging information");
}

// 示例2: 大小轮转文件sink
void example_rotating_file_logger() {
    // 每个文件最大5MB,保留3个历史文件
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "logs/rotating.log", 1024 * 1024 * 5, 3, true);

    auto logger = std::make_shared<spdlog::logger>("rotating_logger", rotating_sink);

    // 设置自定义格式
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

    // 大量日志触发轮转
    for (int i = 0; i < 10000; i++) {
        logger->info("Log message number: {}", i);
    }
}

// 示例3: 多目标分发sink(文件+控制台)
void example_multi_sink_logger() {
    // 创建分发sink
    auto dist_sink = std::make_shared<spdlog::sinks::dist_sink_mt>();

    // 添加子sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multi.log");
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    dist_sink->add_sink(file_sink);
    dist_sink->add_sink(console_sink);

    // 创建logger
    auto logger = std::make_shared<spdlog::logger>("multi_logger", dist_sink);

    // 日志会同时输出到文件和控制台
    logger->info("This message goes to both file and console");
    logger->error("Error message with color on console");
}

// 示例4: 重复过滤sink
void example_dup_filter_logger() {
    // 创建5秒时间窗口的重复过滤器
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_st>(
        std::chrono::seconds(5));

    // 添加实际输出sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    dup_filter->add_sink(console_sink);

    auto logger = std::make_shared<spdlog::logger>("dup_filter_logger", dup_filter);

    // 重复消息会被合并
    logger->warn("Connection failed");  // 输出
    logger->warn("Connection failed");  // 被过滤
    logger->warn("Connection failed");  // 被过滤
    logger->warn("Connection succeeded");  // 5秒后输出 "Skipped 2 duplicate messages.." 然后输出本条
}

// 示例5: 环形缓冲区sink(用于崩溃时dump)
void example_ringbuffer_sink() {
    // 保留最近1000条日志
    auto ringbuffer = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1000);

    auto logger = std::make_shared<spdlog::logger>("ringbuffer_logger", ringbuffer);

    // 正常使用logger
    for (int i = 0; i < 2000; i++) {
        logger->info("Message {}", i);
    }

    // 程序崩溃时dump最近的日志
    auto last_logs = ringbuffer->last_formatted(100);
    for (const auto &log : last_logs) {
        std::cerr << log << std::endl;
    }
}

// 示例6: 自定义回调sink
void example_callback_sink() {
    // 自定义回调函数
    auto callback = [](const spdlog::details::log_msg &msg) {
        // 可以发送到网络、写入自定义格式等
        std::string payload(msg.payload.data(), msg.payload.size());
        printf("Custom callback: [%s] %s\n",
               spdlog::level::to_string_view(msg.level).data(), payload.c_str());
    };

    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(callback);
    auto logger = std::make_shared<spdlog::logger>("callback_logger", callback_sink);

    logger->info("This will trigger the custom callback");
}

// 示例7: ANSI颜色sink (Linux/Mac)
void example_ansicolor_sink() {
    // 创建自动检测终端能力的彩色sink
    auto color_sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>(
        spdlog::color_mode::automatic);

    // 自定义颜色
    color_sink->set_color(spdlog::level::warn, "\033[33m\033[1m");  // 黄色加粗

    auto logger = std::make_shared<spdlog::logger>("color_logger", color_sink);

    logger->info("Green info message");
    logger->warn("Yellow warning message");
    logger->error("Red error message");
}

// 示例8: TCP网络sink (日志聚合)
void example_tcp_sink() {
    // 连接到远程日志服务器
    spdlog::sinks::tcp_sink_config config("logserver.example.com", 9000);
    config.timeout_ms = 5000;

    auto tcp_sink = std::make_shared<spdlog::sinks::tcp_sink_mt>(config);
    auto logger = std::make_shared<spdlog::logger>("tcp_logger", tcp_sink);

    logger->info("Log message sent to remote server");
}

// 示例9: 动态替换sink
void example_dynamic_sink_replacement() {
    auto logger = spdlog::stdout_color_mt("dynamic_logger");

    // 运行时替换sink
    auto new_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/new.log");
    logger->sinks().clear();
    logger->sinks().push_back(new_sink);

    logger->info("Now writing to new file");
}
```

## 5. Implementation Details

**Thread Safety Design**:
- **Mutex Template Parameter**: 所有sink类都通过模板参数`Mutex`实现编译时线程安全策略,多线程环境使用`std::mutex`,单线程环境使用`details::null_mutex`(零开销抽象,所有方法内联为空操作)
- **Lock Granularity**: `base_sink`在public方法级别加锁(log/flush/set_pattern/set_formatter),派生类的`sink_it_()`/`flush_()`已在锁保护下,无需额外加锁
- **Console Mutex**: 控制台sinks使用全局互斥锁`console_mutex`(通过`ConsoleMutex::mutex()`获取),防止多线程输出交错
- **Atomic Level Check**: `sink::should_log()`使用`level_.load(std::memory_order_relaxed)`实现无锁快速检查,relaxed内存序在x86/ARM上开销极低

**Memory Management**:
- **Formatter Ownership**: `formatter_`使用`std::unique_ptr`独占管理,`set_formatter`转移ownership,`set_pattern`构造新`pattern_formatter`
- **Sink Sharing**: `dist_sink`使用`std::shared_ptr`存储子sink,支持多个logger共享同一sink实例
- **Circular Buffer**: `ringbuffer_sink`和`daily/hourly_file_sink`的`filenames_q_`使用`details::circular_q<T>`实现固定容量循环队列,自动覆盖最旧元素
- **String Storage**: `dup_filter_sink`的`last_msg_payload_`使用`std::string`完整拷贝消息内容,避免悬空引用

**Performance Optimizations**:
- **Lazy Initialization**: `current_size_`在`rotating_file_sink`构造时获取一次(昂贵的`stat()`系统调用),后续通过估算更新,只在轮转时同步真实大小
- **Conditional Flush**: `rotating_file_sink`在判断是否轮转时,先检查估算大小`new_size > max_size_`,再检查真实大小`file_helper_.size() > 0`,减少系统调用频率
- **Batch Operations**: `dist_sink::flush_()`和`set_formatter_()`批量操作所有子sink,减少锁竞争
- **Inline Optimization**: `-DSPDLOG_HEADER_ONLY`模式下所有`-inl.h`文件内容内联到调用点,消除函数调用开销

**Error Handling**:
- **Exception Safety**: 文件操作失败(如权限不足、磁盘满)通过`throw_spdlog_ex()`抛出`spdlog_ex`异常,包含错误描述和errno
- **Windows Retry Logic**: `rotating_file_sink::rename_file_()`在Windows上首次重命名失败后等待100ms重试,解决高并发轮转时防病毒软件锁文件问题
- **Graceful Degradation**: `wincolor_sink`检测到`INVALID_HANDLE_VALUE`时静默返回,`stdout_sinks`在无控制台时不输出,`msvc_sink`在无调试器时跳过输出

**Platform-Specific Code**:
- **Windows Console**: 使用`WriteFile`/`WriteConsoleA`/`WriteConsoleW` API替代标准库`fwrite`,避免CRLF转换问题(issue #1675),通过`_get_osfhandle`获取FILE*对应的HANDLE
- **Color Detection**: Unix下`ansicolor_sink`通过`details::os::in_terminal()`检测TERM环境变量和`isatty()`,通过`details::os::is_color_terminal()`检测COLORTERM环境变量;Windows下`wincolor_sink`使用`GetConsoleMode`检测真实控制台
- **UTF-8 Support**: Windows控制台支持`SPDLOG_UTF8_TO_WCHAR_CONSOLE`宏,通过`os::utf8_to_wstrbuf()`转换为UTF-16后使用`WriteConsoleW`输出
- **File Rotation**: Windows重命名操作先调用`remove()`删除target文件,解决`std::rename`在目标存在时失败的问题

**Design Patterns Used**:
- **Strategy Pattern**: sink接口定义日志输出策略,30多个具体策略类可互换
- **Template Method Pattern**: `base_sink`定义算法骨架(加锁→调用虚函数→解锁),派生类实现具体步骤
- **CRTP (Curiously Recurring Template Pattern)**: `base_sink<Mutex>`使用模板实现静态多态,避免虚函数开销
- **Factory Pattern**: 每个sink类型提供对应的`xxx_logger_mt/st`工厂函数,封装logger创建逻辑
- **Composite Pattern**: `dist_sink`管理多个子sink,形成sink树形结构,支持递归组合
- **Decorator Pattern**: `dup_filter_sink`继承`dist_sink`,在转发前添加重复过滤逻辑

**Third-Party Integrations**:
- **Android Sink**: 使用`__android_log_write`或`__android_log_buf_write`(非默认buffer时),映射spdlog级别到Android日志优先级,支持EAGAIN错误重试(最多`SPDLOG_ANDROID_RETRIES`次)
- **Kafka Sink**: 依赖`librdkafka`,使用`RdKafka::Producer`/`RdKafka::Topic`/`RdKafka::Conf`API,构造时初始化producer和topic,析构时调用`producer_->flush(timeout_ms)`
- **MongoDB Sink**: 依赖`mongocxx`,使用`mongocxx::client`/`mongocxx::instance`,将日志消息转换为BSON文档:`{timestamp, level, level_num, message, logger_name, thread_id}`,通过`insert_one(doc.view())`写入
- **Qt Sink**: 使用`QMetaObject::invokeMethod`线程安全调用Qt对象的元方法(如`append`、`appendPlainText`),`qt_color_sink`支持UTF-8和max_lines限制,通过`QTextCursor`操作`QTextEdit`文档
