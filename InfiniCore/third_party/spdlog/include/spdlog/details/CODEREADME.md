# spdlog::details 内部实现模块文档

## 概述

`spdlog::details` 是 spdlog 日志库的核心内部实现模块，提供了底层数据结构、并发原语、操作系统抽象、日志消息处理、线程池、文件操作、网络客户端等关键基础设施。该模块不直接暴露给最终用户，而是为上层 spdlog API 提供支撑。

## 1. 模块结构

- **`circular_q.h`**: 基于循环缓冲区的无锁队列实现
- **`mpmc_blocking_q.h`**: 多生产者多消费者阻塞队列
- **`log_msg.h` / `log_msg-inl.h`**: 日志消息数据结构
- **`log_msg_buffer.h` / `log_msg_buffer-inl.h`**: 带缓冲的日志消息（支持消息生命周期延长）
- **`backtracer.h` / `backtracer-inl.h`**: 日志回溯功能（存储最近的 N 条日志用于调试）
- **`registry.h` / `registry-inl.h`**: 全局 logger 注册表（单例模式）
- **`thread_pool.h` / `thread_pool-inl.h`**: 异步日志线程池
- **`file_helper.h` / `file_helper-inl.h`**: 文件操作辅助类（打开、写入、刷新、同步）
- **`os.h` / `os-inl.h`**: 操作系统抽象层（跨平台文件、时间、线程、目录操作）
- **`fmt_helper.h`**: 格式化辅助函数（数字填充、时间处理）
- **`periodic_worker.h` / `periodic_worker-inl.h`**: 周期性任务执行器（RAII 管理的后台线程）
- **`tcp_client.h` / `tcp_client-windows.h`**: TCP 客户端（支持超时连接、非阻塞模式）
- **`udp_client.h` / `udp_client-windows.h`**: UDP 客户端
- **`null_mutex.h`**: 空 Mutex（零开销的单线程优化）
- **`console_globals.h`**: 控制台全局 Mutex 管理器
- **`synchronous_factory.h`**: 同步 logger 工厂
- **`windows_include.h`**: Windows 头文件包含宏定义

## 2. 核心类

### 2.1 `circular_q<T>`
- **位置**: `circular_q.h`
- **主要功能**: 基于 `std::vector` 的固定大小循环队列，支持自动覆盖最旧元素
- **关键成员**:
  - `max_items_`: 容量（实际分配 `max_items_ + 1`，保留一个哨兵位置用于区分满和空）
  - `head_`: 队头索引（读取位置）
  - `tail_`: 队尾索引（写入位置）
  - `overrun_counter_`: 覆盖计数器
  - `v_`: 底层存储向量
- **核心方法**:
  - `push_back(T &&item)`: O(1) 入队，满时自动覆盖最旧元素并递增 `overrun_counter_`
  - `front()`: O(1) 获取队头元素引用
  - `pop_front()`: O(1) 出队
  - `size()`: O(1) 返回元素数量（处理环形回绕）
  - `at(size_t i)`: O(1) 随机访问，支持索引越界断言
  - `empty()`: O(1) 判空（`head_ == tail_`）
  - `full()`: O(1) 判满（`(tail_ + 1) % max_items_ == head_`）
- **生命周期**: 支持移动语义（移动后源对象进入 disabled 状态，`max_items_ = 0`）

### 2.2 `mpmc_blocking_queue<T>`
- **位置**: `mpmc_blocking_q.h`
- **主要功能**: 多生产者多消费者线程安全阻塞队列，基于条件变量和互斥锁
- **关键成员**:
  - `q_`: 底层 `circular_q<T>` 容器
  - `queue_mutex_`: 保护队列的 `std::mutex`
  - `push_cv_`: 队列非空条件变量（唤醒消费者）
  - `pop_cv_`: 队列非满条件变量（唤醒生产者）
  - `discard_counter_`: `std::atomic<size_t>` 满时丢弃消息计数
- **核心方法**:
  - `enqueue(T &&item)`: 阻塞式入队，队列满时等待直到有空间
  - `enqueue_nowait(T &&item)`: 非阻塞入队，满时覆盖最旧元素
  - `enqueue_if_have_room(T &&item)`: 条件入队，满时丢弃新消息并递增 `discard_counter_`
  - `dequeue(T &popped_item)`: 阻塞式出队，队列空时等待
  - `dequeue_for(T &popped_item, milliseconds timeout)`: 超时出队
  - `overrun_counter()`: 返回覆盖计数
  - `discard_counter()`: 返回丢弃计数
- **并发保证**: 使用 `std::unique_lock` + 条件变量，支持 MinGW 特殊处理（避免死锁）

### 2.3 `log_msg`
- **位置**: `log_msg.h`
- **主要功能**: 不可变日志消息数据结构，持有 `string_view_t`（引用外部栈数据）
- **关键成员**:
  - `logger_name`: `string_view_t` logger 名称
  - `level`: `level::level_enum` 日志级别
  - `time`: `log_clock::time_point` 时间戳
  - `thread_id`: `size_t` 线程 ID（通过 `os::thread_id()` 获取）
  - `source`: `source_loc` 源码位置（文件名、行号、函数名）
  - `payload`: `string_view_t` 格式化后的日志文本
  - `color_range_start` / `color_range_end`: `mutable size_t` 颜色标记范围（用于终端着色）
- **构造函数**: 三种重载（完整参数、自动时间戳、默认源码位置）
- **生命周期**: 轻量级引用语义，构造时自动捕获线程 ID 和时间戳

### 2.4 `log_msg_buffer`
- **位置**: `log_msg_buffer.h` / `log_msg_buffer-inl.h`
- **主要功能**: 继承自 `log_msg`，内部持有 `memory_buf_t` 缓冲区，将引用数据转为拥有数据
- **关键成员**:
  - `buffer`: `memory_buf_t`（即 `std::vector<char>`）存储 logger_name 和 payload 的副本
- **核心方法**:
  - `update_string_views()`: 将 `logger_name` 和 `payload` 的 `string_view_t` 重新指向 `buffer`
- **使用场景**: 需要延长日志消息生命周期时（如存入 `backtracer` 或异步队列）
- **构造逻辑**: 从 `log_msg` 复制时，将 `logger_name` 和 `payload` 追加到 `buffer`，然后更新视图指针

### 2.5 `backtracer`
- **位置**: `backtracer.h` / `backtracer-inl.h`
- **主要功能**: 启用时存储最近的 N 条日志消息，用于错误/警告发生时回溯上下文
- **关键成员**:
  - `enabled_`: `std::atomic<bool>` 启用标志
  - `messages_`: `circular_q<log_msg_buffer>` 消息队列
  - `mutex_`: `std::mutex` 保护队列操作
- **核心方法**:
  - `enable(size_t size)`: 创建指定大小的循环队列，设置 `enabled_ = true`
  - `disable()`: 清空队列，设置 `enabled_ = false`
  - `push_back(const log_msg &msg)`: 如果启用，将消息转为 `log_msg_buffer` 并入队
  - `foreach_pop(std::function<void(const log_msg &)> fun)`: 弹出所有消息并应用回调
  - `empty()`: 检查队列是否为空
- **线程安全**: 所有操作都加锁，`enabled_` 使用 `memory_order_relaxed`

### 2.6 `registry`
- **位置**: `registry.h` / `registry-inl.h`
- **主要功能**: 全局 logger 注册表（单例模式），管理所有命名 logger、默认 logger、线程池、全局格式化器、刷新策略
- **关键成员**:
  - `loggers_`: `std::unordered_map<std::string, std::shared_ptr<logger>>` logger 映射
  - `default_logger_`: `std::shared_ptr<logger>` 默认 logger（用于 `spdlog::info()` 等全局函数）
  - `tp_`: `std::shared_ptr<thread_pool>` 异步线程池
  - `formatter_`: `std::unique_ptr<formatter>` 全局格式化器
  - `global_log_level_`: `level::level_enum` 全局日志级别
  - `flush_level_`: `level::level_enum` 自动刷新级别
  - `periodic_flusher_`: `std::unique_ptr<periodic_worker>` 周期性刷新器
  - `automatic_registration_`: `bool` 是否自动注册新创建的 logger
  - `backtrace_n_messages_`: `size_t` 全局回溯消息数量
  - `log_levels_`: `log_levels`（即 `std::unordered_map<std::string, level::level_enum>`）按 logger 名称配置的级别
  - `logger_map_mutex_`: `std::mutex` 保护 `loggers_` 和 `default_logger_`
  - `tp_mutex_`: `std::recursive_mutex` 保护线程池
  - `flusher_mutex_`: `std::mutex` 保护周期性刷新器
- **核心方法**:
  - `register_logger(std::shared_ptr<logger>)`: 注册命名 logger，名称冲突抛异常
  - `register_or_replace(std::shared_ptr<logger>)`: 注册或替换同名 logger
  - `initialize_logger(std::shared_ptr<logger>)`: 初始化 logger（克隆全局格式化器、设置级别、刷新策略、回溯、错误处理器），根据 `automatic_registration_` 决定是否注册
  - `get(const std::string &)`: 获取命名 logger，不存在返回 `nullptr`
  - `default_logger()`: 返回默认 logger 的 `shared_ptr`
  - `get_default_raw()`: 返回默认 logger 的原始指针（用于 `spdlog::info()` 等热路径，但需注意并发问题）
  - `set_default_logger(std::shared_ptr<logger>)`: 设置默认 logger，同时注册到映射表
  - `set_tp(std::shared_ptr<thread_pool>)`: 设置全局线程池
  - `set_formatter(std::unique_ptr<formatter>)`: 设置全局格式化器，所有现有 logger 的 sink 都会克隆一份
  - `set_level(level::level_enum)`: 设置所有 logger 的级别，更新 `global_log_level_`
  - `flush_on(level::level_enum)`: 设置所有 logger 的自动刷新级别
  - `flush_every(chrono::duration)`: 启动周期性刷新器（使用 `periodic_worker`）
  - `set_error_handler(err_handler)`: 设置所有 logger 的错误处理器
  - `enable_backtrace(size_t)`: 启用所有 logger 的回溯功能
  - `apply_all(std::function<void(std::shared_ptr<logger>)>)`: 对所有 logger 应用回调
  - `flush_all()`: 刷新所有 logger
  - `drop(const std::string &)`: 移除指定 logger，如果是默认 logger 则重置
  - `drop_all()`: 清空所有 logger 并重置默认 logger
  - `shutdown()`: 停止周期性刷新器，清空 logger，重置线程池
  - `set_levels(log_levels, level::level_enum *)`: 批量设置 logger 级别（支持全局级别）
  - `apply_logger_env_levels(std::shared_ptr<logger>)`: 对单个 logger 应用已配置的级别
- **生命周期**: 单例模式，构造时创建默认 logger（Windows 用 `wincolor_stdout_sink_mt`，其他平台用 `ansicolor_stdout_sink_mt`），析构时自动调用 `shutdown()`

### 2.7 `thread_pool`
- **位置**: `thread_pool.h` / `thread_pool-inl.h`
- **主要功能**: 异步日志线程池，管理多个工作线程消费 `mpmc_blocking_queue<async_msg>`
- **关键成员**:
  - `q_`: `mpmc_blocking_queue<async_msg>` 消息队列
  - `threads_`: `std::vector<std::thread>` 工作线程
- **核心枚举**:
  - `async_msg_type`: `log`（日志消息）、`flush`（刷新请求）、`terminate`（终止信号）
- **核心结构**:
  - `async_msg`: 继承自 `log_msg_buffer`，包含 `msg_type` 和 `worker_ptr`（指向发起请求的 `async_logger`）
- **核心方法**:
  - `thread_pool(size_t q_max_items, size_t threads_n, ...)`: 构造函数，创建指定数量的工作线程，每个线程执行 `worker_loop_()`
  - `post_log(async_logger_ptr &&, const log_msg &, async_overflow_policy)`: 发送日志消息到队列
  - `post_flush(async_logger_ptr &&, async_overflow_policy)`: 发送刷新请求到队列
  - `overrun_counter()`: 返回队列覆盖计数
  - `discard_counter()`: 返回队列丢弃计数
  - `queue_size()`: 返回队列当前大小
- **溢出策略**:
  - `block`: 队列满时阻塞等待（`q_.enqueue()`）
  - `overrun_oldest`: 队列满时覆盖最旧消息（`q_.enqueue_nowait()`）
  - `discard_new`: 队列满时丢弃新消息（`q_.enqueue_if_have_room()`）
- **工作线程逻辑**:
  - `worker_loop_()`: 循环调用 `process_next_msg_()`，返回 `false` 时退出
  - `process_next_msg_()`: 从队列出队消息，根据类型分发：
    - `log`: 调用 `worker_ptr->backend_sink_it_(msg)`
    - `flush`: 调用 `worker_ptr->backend_flush_()`
    - `terminate`: 返回 `false` 终止线程
- **生命周期**: 析构时向队列发送与线程数相等的 `terminate` 消息，然后 `join()` 所有线程

### 2.8 `file_helper`
- **位置**: `file_helper.h` / `file_helper-inl.h`
- **主要功能**: 文件操作辅助类，处理文件打开、写入、刷新、同步、大小查询
- **关键成员**:
  - `fd_`: `std::FILE *` 文件句柄
  - `filename_`: `filename_t`（`std::string` 或 `std::wstring`）当前打开的文件名
  - `event_handlers_`: `file_event_handlers` 事件回调（`before_open`、`after_open`、`before_close`、`after_close`）
  - `open_tries_`: `const int = 5` 打开重试次数
  - `open_interval_`: `const unsigned int = 10` 重试间隔（毫秒）
- **核心方法**:
  - `open(const filename_t &fname, bool truncate)`: 打开文件，自动创建父目录，`truncate=true` 时先清空，重试机制（5 次，间隔 10ms），失败抛 `spdlog_ex`
  - `reopen(bool truncate)`: 重新打开当前文件名（用于文件被外部移动/截断后恢复）
  - `flush()`: 调用 `std::fflush(fd_)`，失败抛异常
  - `sync()`: 调用 `os::fsync(fd_)`，将数据刷入磁盘，失败抛异常
  - `close()`: 关闭文件，触发 `before_close` 和 `after_close` 事件
  - `write(const memory_buf_t &buf)`: 调用 `os::fwrite_bytes()` 写入数据
  - `size()`: 调用 `os::filesize(fd_)` 获取文件大小
  - `split_by_extension(const filename_t &fname)`: 静态方法，分割文件路径和扩展名（处理隐藏文件、多级目录）
- **线程安全**: 非线程安全，外部需同步
- **平台差异**: Windows 支持 Unicode 文件名（`SPDLOG_WCHAR_FILENAMES`），支持子进程继承控制（`SPDLOG_PREVENT_CHILD_FD`）

### 2.9 `os` 命名空间
- **位置**: `os.h` / `os-inl.h`
- **主要功能**: 跨平台操作系统抽象层，封装文件、时间、线程、目录、终端、环境变量等系统调用
- **核心函数**:
  - `now()`: 返回当前时间点（Linux 支持 `CLOCK_REALTIME_COARSE` 优化）
  - `localtime(const std::time_t &)`: 线程安全的本地时间转换（Windows 用 `localtime_s`，Unix 用 `localtime_r`）
  - `gmtime(const std::time_t &)`: 线程安全的 UTC 时间转换
  - `fopen_s(FILE **fp, const filename_t &, const filename_t &)`: 跨平台 `fopen`，支持子进程继承控制（`O_CLOEXEC` 或 `SetHandleInformation`）
  - `remove(const filename_t &)`: 删除文件
  - `remove_if_exists(const filename_t &)`: 存在则删除
  - `rename(const filename_t &, const filename_t &)`: 重命名文件
  - `path_exists(const filename_t &)`: 检查路径是否存在（文件或目录）
  - `filesize(FILE *f)`: 获取文件大小（Windows 用 `_filelengthi64`，Unix 用 `fstat`/`fstat64`）
  - `utc_minutes_offset(const std::tm &)`: 计算 UTC 偏移量（分钟），支持 SunOS/AIX 缺失 `tm_gmtoff` 字段的回退计算
  - `_thread_id()`: 获取线程 ID（Windows 用 `GetCurrentThreadId`，Linux 用 `syscall(SYS_gettid)`，macOS 用 `pthread_threadid_np`，其他用 `std::hash<std::thread::id>()`）
  - `thread_id()`: 从线程局部缓存返回线程 ID（`SPDLOG_NO_TLS` 时每次调用 `_thread_id()`）
  - `sleep_for_millis(unsigned int)`: 跨平台睡眠（Windows 用 `Sleep`，Unix 用 `std::this_thread::sleep_for`）
  - `filename_to_str(const filename_t &)`: 宽文件名转 UTF-8 字符串（Windows 且 `SPDLOG_WCHAR_FILENAMES` 时使用 `wstr_to_utf8buf`）
  - `pid()`: 获取进程 ID（Windows 用 `GetCurrentProcessId`，Unix 用 `getpid`）
  - `is_color_terminal()`: 检测终端是否支持颜色（Windows 总是返回 `true`，Unix 检查 `COLORTERM` 环境变量和 `TERM` 值）
  - `in_terminal(FILE *file)`: 检测文件是否关联到终端（Windows 用 `_isatty`，Unix 用 `isatty`）
  - `wstr_to_utf8buf(wstring_view_t, memory_buf_t &)`: Windows 宽字符转 UTF-8（使用 `WideCharToMultiByte`）
  - `utf8_to_wstrbuf(string_view_t, wmemory_buf_t &)`: Windows UTF-8 转宽字符（使用 `MultiByteToWideChar`）
  - `dir_name(const filename_t &path)`: 提取目录路径（`"abc/file" => "abc"`，`"abc" => ""`）
  - `create_dir(const filename_t &path)`: 递归创建目录，已存在则成功
  - `getenv(const char *field)`: 获取环境变量（非线程安全，Windows UWP 不支持）
  - `fsync(FILE *fp)`: 刷新文件到磁盘（Windows 用 `FlushFileBuffers`，Unix 用 `fsync`）
  - `fwrite_bytes(const void *, size_t, FILE *fp)`: 非锁定写入（`SPDLOG_FWRITE_UNLOCKED` 时用 `fwrite_unlocked` 或 `_fwrite_nolock`）
- **常量定义**:
  - `SPDLOG_EOL`: 行结束符（Windows `"\r\n"`，Unix `"\n"`）
  - `SPDLOG_FOLDER_SEPS`: 目录分隔符（Windows `"\"`，Unix `"/"`）

### 2.10 `fmt_helper` 命名空间
- **位置**: `fmt_helper.h`
- **主要功能**: 格式化辅助函数，优化数字填充和时间处理
- **核心函数**:
  - `append_string_view(string_view_t, memory_buf_t &)`: 追加字符串视图到缓冲区
  - `append_int(T n, memory_buf_t &)`: 追加整数到缓冲区（`SPDLOG_USE_STD_FORMAT` 时用 `std::to_chars`，否则用 `fmt::format_int`）
  - `count_digits(T n)`: 计算整数位数（`SPDLOG_USE_STD_FORMAT` 时用回退算法，否则用 `fmt::detail::count_digits`）
  - `count_digits_fallback(T n)`: 快速位数计算（每 4 次除法处理 4 位数字，来自 Alexandrescu 优化技巧）
  - `pad2(int n, memory_buf_t &)`: 填充到 2 位（0-99 直接计算，其他用 `fmt::format_to`）
  - `pad_uint(T n, unsigned int width, memory_buf_t &)`: 无符号整数左零填充到指定宽度
  - `pad3(T n, memory_buf_t &)`: 填充到 3 位（仅支持无符号，0-999 直接计算）
  - `pad6(T n, memory_buf_t &)`: 填充到 6 位
  - `pad9(T n, memory_buf_t &)`: 填充到 9 位
  - `time_fraction<ToDuration>(log_clock::time_point)`: 提取秒的小数部分（如毫秒、微秒、纳秒）

### 2.11 `periodic_worker`
- **位置**: `periodic_worker.h` / `periodic_worker-inl.h`
- **主要功能**: RAII 管理的周期性任务执行器，创建后台线程按固定间隔执行回调
- **关键成员**:
  - `active_`: `bool` 活跃标志（间隔为 0 时不创建线程）
  - `worker_thread_`: `std::thread` 工作线程
  - `mutex_`: `std::mutex` 保护 `active_`
  - `cv_`: `std::condition_variable` 用于超时等待
- **核心方法**:
  - `periodic_worker(std::function<void()> callback, chrono::duration interval)`: 构造函数，创建线程执行循环：`cv_.wait_for(interval, [this]{ return !active_; })`，超时后执行 `callback()`，如果 `active_` 为 `false` 则退出
  - `~periodic_worker()`: 析构函数，设置 `active_ = false`，通知 `cv_`，`join()` 线程
  - `get_thread()`: 返回工作线程引用
- **生命周期**: RAII 模式，线程与对象生命周期绑定

### 2.12 `tcp_client`
- **位置**: `tcp_client.h`（Unix）/ `tcp_client-windows.h`（Windows）
- **主要功能**: TCP 客户端，支持超时连接、非阻塞模式、TCP_NODELAY、SIGPIPE 防护
- **关键成员**:
  - `socket_`: `int`（Unix）/ `SOCKET`（Windows）套接字句柄
- **核心方法**:
  - `connect(const std::string &host, int port, int timeout_ms = 0)`: 连接到指定主机和端口
    - 使用 `getaddrinfo` 解析主机名（支持 IPv4/IPv6）
    - 尝试所有返回地址直到成功
    - `timeout_ms > 0` 时启用超时模式：
      - 设置套接字为非阻塞（`fcntl` / `ioctlsocket`）
      - 调用 `connect`
      - 使用 `select` 等待可写或超时
      - 检查 `SO_ERROR` 判断连接是否成功
      - 恢复阻塞模式
    - 设置 `SO_RCVTIMEO` 和 `SO_SNDTIMEO`
    - 设置 `TCP_NODELAY`（禁用 Nagle 算法）
    - Unix 设置 `SO_NOSIGPIPE` 或使用 `MSG_NOSIGNAL`（防止 SIGPIPE）
  - `send(const char *data, size_t n_bytes)`: 循环发送直到全部数据发送完毕
    - 使用 `send` 系统调用
    - 处理部分发送情况
    - 错误时关闭连接并抛异常
  - `close()`: 关闭套接字
  - `is_connected()`: 检查是否已连接
  - `fd()`: 返回套接字句柄
- **平台差异**:
  - Unix: 使用 `socket`、`connect`、`select`、`fcntl`、`close`、`send`
  - Windows: 使用 `WSAStartup`、`socket`、`connect`、`select`、`ioctlsocket`、`closesocket`、`send`，错误处理用 `WSAGetLastError` 和 `FormatMessageA`

### 2.13 `udp_client`
- **位置**: `udp_client.h`（Unix）/ `udp_client-windows.h`（Windows）
- **主要功能**: UDP 客户端，RAII 管理，支持设置发送缓冲区大小
- **关键成员**:
  - `socket_`: `int`（Unix）/ `SOCKET`（Windows）套接字句柄
  - `sockAddr_` / `addr_`: `sockaddr_in` 目标地址
  - `TX_BUFFER_SIZE`: `static constexpr int = 1024 * 10` 发送缓冲区大小（10KB）
- **核心方法**:
  - `udp_client(const std::string &host, uint16_t port)`: 构造函数
    - 创建 `SOCK_DGRAM` 套接字
    - 设置 `SO_SNDBUF` 为 10KB
    - 使用 `inet_aton`（Unix）/ `InetPtonA`（Windows）转换 IP 地址
    - 设置端口（`htons`）
  - `send(const char *data, size_t n_bytes)`: 调用 `sendto` 发送数据
  - `fd()`: 返回套接字句柄
  - `cleanup_()`: 关闭套接字并清理（Unix 仅关闭，Windows 还调用 `WSACleanup`）
- **平台差异**: Windows 需要 `WSAStartup` 初始化，析构时调用 `WSACleanup`

### 2.14 `null_mutex`
- **位置**: `null_mutex.h`
- **主要功能**: 空 Mutex（零开销的假 Mutex，用于单线程优化）
- **关键方法**:
  - `lock()`: 空操作
  - `unlock()`: 空操作
- **使用场景**: 当 logger 确定为单线程时，可用 `null_mutex` 替代 `std::mutex` 避免同步开销

### 2.15 `null_atomic_int`
- **位置**: `null_mutex.h`
- **主要功能**: 模拟原子操作的普通整数（零开销）
- **关键成员**:
  - `value`: `int` 普通整数
- **关键方法**:
  - `load(memory_order)`: 直接返回 `value`
  - `store(int, memory_order)`: 直接赋值
  - `exchange(int, memory_order)`: 使用 `std::swap` 交换并返回旧值

### 2.16 `console_mutex` / `console_nullmutex`
- **位置**: `console_globals.h`
- **主要功能**: 控制台全局 Mutex 管理器（函数局部静态变量模式）
- **关键方法**:
  - `static mutex_t &mutex()`: 返回静态 `std::mutex`（或 `null_mutex`）引用
- **使用场景**: 多个 sink 共享同一个控制台时避免输出交错

### 2.17 `synchronous_factory`
- **位置**: `synchronous_factory.h`
- **主要功能**: 默认 logger 工厂，创建同步 logger（非异步）
- **核心方法**:
  - `create<Sink, ...SinkArgs>(std::string logger_name, SinkArgs &&...args)`: 创建 sink 和 logger，调用 `registry::instance().initialize_logger()`
- **使用场景**: 作为 `spdlog::create` 等函数的默认工厂类型

## 3. API 接口

### 3.1 全局函数（通过 `registry` 单例暴露）

```cpp
// 获取全局注册表实例
registry &registry::instance();

// 注册和管理 logger
void registry::register_logger(std::shared_ptr<logger> new_logger);
void registry::register_or_replace(std::shared_ptr<logger> new_logger);
void registry::initialize_logger(std::shared_ptr<logger> new_logger);
std::shared_ptr<logger> registry::get(const std::string &logger_name);
std::shared_ptr<logger> registry::default_logger();
logger *registry::get_default_raw(); // 热路径优化
void registry::set_default_logger(std::shared_ptr<logger> new_default_logger);

// 全局配置
void registry::set_tp(std::shared_ptr<thread_pool> tp);
void registry::set_formatter(std::unique_ptr<formatter> formatter);
void registry::set_level(level::level_enum log_level);
void registry::flush_on(level::level_enum log_level);
void registry::set_error_handler(err_handler handler);

// 批量操作
void registry::apply_all(std::function<void(const std::shared_ptr<logger>)> fun);
void registry::flush_all();
void registry::drop(const std::string &logger_name);
void registry::drop_all();
void registry::shutdown();

// 线程池操作
std::shared_ptr<thread_pool> registry::get_tp();
std::recursive_mutex &registry::tp_mutex();

// 周期性刷新
template <typename Rep, typename Period>
void registry::flush_every(std::chrono::duration<Rep, Period> interval);
```

### 3.2 操作系统抽象（`os` 命名空间）

```cpp
// 时间操作
log_clock::time_point os::now() SPDLOG_NOEXCEPT;
std::tm os::localtime(const std::time_t &time_tt) SPDLOG_NOEXCEPT;
std::tm os::localtime() SPDLOG_NOEXCEPT;
std::tm os::gmtime(const std::time_t &time_tt) SPDLOG_NOEXCEPT;
std::tm os::gmtime() SPDLOG_NOEXCEPT;

// 文件操作
bool os::fopen_s(FILE **fp, const filename_t &filename, const filename_t &mode);
int os::remove(const filename_t &filename) SPDLOG_NOEXCEPT;
int os::remove_if_exists(const filename_t &filename) SPDLOG_NOEXCEPT;
int os::rename(const filename_t &filename1, const filename_t &filename2) SPDLOG_NOEXCEPT;
bool os::path_exists(const filename_t &filename) SPDLOG_NOEXCEPT;
size_t os::filesize(FILE *f);
bool os::fsync(FILE *fp);
bool os::fwrite_bytes(const void *ptr, size_t n_bytes, FILE *fp);

// 目录操作
filename_t os::dir_name(const filename_t &path);
bool os::create_dir(const filename_t &path);

// 线程操作
size_t os::_thread_id() SPDLOG_NOEXCEPT;
size_t os::thread_id() SPDLOG_NOEXCEPT;
void os::sleep_for_millis(unsigned int milliseconds) SPDLOG_NOEXCEPT;

// 系统信息
int os::pid() SPDLOG_NOEXCEPT;
int os::utc_minutes_offset(const std::tm &tm = details::os::localtime());
std::string os::getenv(const char *field);

// 终端检测
bool os::is_color_terminal() SPDLOG_NOEXCEPT;
bool os::in_terminal(FILE *file) SPDLOG_NOEXCEPT;

// 宽字符转换（Windows）
#if defined(SPDLOG_WCHAR_TO_UTF8_SUPPORT) && defined(_WIN32)
void os::wstr_to_utf8buf(wstring_view_t wstr, memory_buf_t &target);
void os::utf8_to_wstrbuf(string_view_t str, wmemory_buf_t &target);
#endif
```

### 3.3 格式化辅助（`fmt_helper` 命名空间）

```cpp
// 字符串和整数追加
void fmt_helper::append_string_view(spdlog::string_view_t view, memory_buf_t &dest);
template <typename T>
void fmt_helper::append_int(T n, memory_buf_t &dest);

// 位数计算
template <typename T>
unsigned int fmt_helper::count_digits(T n);
template <typename T>
unsigned int fmt_helper::count_digits_fallback(T n);

// 数字填充
void fmt_helper::pad2(int n, memory_buf_t &dest);
template <typename T>
void fmt_helper::pad_uint(T n, unsigned int width, memory_buf_t &dest);
template <typename T>
void fmt_helper::pad3(T n, memory_buf_t &dest);
template <typename T>
void fmt_helper::pad6(T n, memory_buf_t &dest);
template <typename T>
void fmt_helper::pad9(T n, memory_buf_t &dest);

// 时间分数提取
template <typename ToDuration>
ToDuration fmt_helper::time_fraction(log_clock::time_point tp);
```

## 4. 使用示例

### 4.1 使用 `circular_q` 构建固定大小缓存

```cpp
#include <spdlog/details/circular_q.h>

// 创建最多存储 100 个整数的循环队列
spdlog::details::circular_q<int> queue(100);

// 入队（满时自动覆盖最旧元素）
queue.push_back(42);
queue.push_back(100);

// 访问队头元素
if (!queue.empty()) {
    int first = queue.front(); // O(1)
    queue.pop_front();         // O(1)
}

// 随机访问
for (size_t i = 0; i < queue.size(); ++i) {
    std::cout << queue.at(i) << "\n";
}

// 检查覆盖计数
std::cout << "Overrun: " << queue.overrun_counter() << "\n";
queue.reset_overrun_counter();
```

### 4.2 使用 `mpmc_blocking_queue` 构建生产者-消费者系统

```cpp
#include <spdlog/details/mpmc_blocking_q.h>
#include <thread>
#include <vector>

spdlog::details::mpmc_blocking_queue<int> queue(1024);

// 生产者线程
std::thread producer([&]() {
    for (int i = 0; i < 1000; ++i) {
        // 阻塞式入队（队列满时等待）
        queue.enqueue(i);
    }
});

// 消费者线程
std::thread consumer([&]() {
    int value;
    while (true) {
        // 阻塞式出队（队列空时等待）
        queue.dequeue(value);
        if (value == 999) break;
        std::cout << "Consumed: " << value << "\n";
    }
});

producer.join();
consumer.join();

// 查看统计信息
std::cout << "Overrun: " << queue.overrun_counter() << "\n";
std::cout << "Discard: " << queue.discard_counter() << "\n";
```

### 4.3 使用 `backtracer` 捕获最近的日志

```cpp
#include <spdlog/details/backtracer.h>
#include <spdlog/logger.h>

spdlog::details::backtracer tracer;

// 启用回溯，存储最近 64 条日志
tracer.enable(64);

// 模拟日志记录
for (int i = 0; i < 100; ++i) {
    spdlog::details::log_msg msg("my_logger", spdlog::level::info, "Log " + std::to_string(i));
    tracer.push_back(msg);
}

// 发生错误时，弹出所有回溯消息并处理
std::cout << "Dumping backtrace:\n";
tracer.foreach_pop([](const spdlog::details::log_msg &msg) {
    std::cout << msg.payload.data() << "\n";
});

tracer.disable();
```

### 4.4 使用 `thread_pool` 构建异步任务系统

```cpp
#include <spdlog/details/thread_pool.h>

// 创建线程池：队列容量 4096，4 个工作线程
spdlog::details::thread_pool pool(4096, 4);

// 自定义任务类型（模拟 async_logger）
struct async_msg {
    enum class Type { LOG, FLUSH, TERMINATE } type;
    std::string data;
};

// 模拟发送日志任务
auto logger = std::make_shared<spdlog::logger>("async_logger");
spdlog::details::log_msg msg("async_logger", spdlog::level::info, "Async log message");

// 使用 overrun_oldest 策略（满时覆盖最旧消息）
pool.post_log(logger, msg, spdlog::async_overflow_policy::overrun_oldest);

// 发送刷新请求
pool.post_flush(logger, spdlog::async_overflow_policy::block);

// 查看统计
std::cout << "Queue size: " << pool.queue_size() << "\n";
std::cout << "Overrun: " << pool.overrun_counter() << "\n";
std::cout << "Discard: " << pool.discard_counter() << "\n";

// 析构时自动停止所有工作线程
```

### 4.5 使用 `file_helper` 管理日志文件

```cpp
#include <spdlog/details/file_helper.h>

spdlog::details::file_helper file_helper;

// 打开文件（自动创建父目录，失败重试 5 次，间隔 10ms）
file_helper.open("logs/myapp.log", false); // false = 追加模式

// 写入数据
spdlog::memory_buf_t buf;
buf.append("Log message\n", 12);
file_helper.write(buf);

// 刷新到操作系统缓冲区
file_helper.flush();

// 同步到磁盘（确保数据落盘）
file_helper.sync();

// 获取文件大小
std::cout << "File size: " << file_helper.size() << " bytes\n";

// 分离文件名和扩展名
auto [base, ext] = spdlog::details::file_helper::split_by_extension("logs/app.log");
std::cout << "Base: " << base << ", Ext: " << ext << "\n"; // "logs/app", ".log"

// 关闭文件
file_helper.close();
```

### 4.6 使用 `registry` 管理 logger

```cpp
#include <spdlog/details/registry.h>

auto &reg = spdlog::details::registry::instance();

// 创建并注册 logger
auto logger = std::make_shared<spdlog::logger>("my_logger");
reg.register_logger(logger);

// 获取 logger
auto retrieved = reg.get("my_logger");
if (retrieved) {
    retrieved->info("Retrieved logger");
}

// 设置默认 logger
reg.set_default_logger(retrieved);

// 设置全局格式化器
auto formatter = std::make_unique<spdlog::pattern_formatter>("[%Y-%m-%d %H:%M:%S] %v");
reg.set_formatter(std::move(formatter)); // 所有现有 logger 都会克隆一份

// 设置全局日志级别
reg.set_level(spdlog::level::debug);

// 启用所有 logger 的回溯（存储最近 32 条）
reg.enable_backtrace(32);

// 周期性刷新（每 5 秒刷新一次所有 logger）
reg.flush_every(std::chrono::seconds(5));

// 对所有 logger 应用操作
reg.apply_all([](const std::shared_ptr<spdlog::logger> &logger) {
    logger->flush();
});

// 移除指定 logger
reg.drop("my_logger");

// 清空所有 logger
reg.drop_all();

// 关闭并清理资源（停止周期性刷新器，重置线程池）
reg.shutdown();
```

### 4.7 使用 `periodic_worker` 执行周期性任务

```cpp
#include <spdlog/details/periodic_worker.h>
#include <iostream>

int counter = 0;

// 创建周期性任务：每 1 秒执行一次回调
spdlog::details::periodic_worker worker(
    [&counter]() {
        std::cout << "Task executed: " << ++counter << "\n";
        std::cout << "Flushing logs...\n";
    },
    std::chrono::seconds(1)
);

// 主线程继续工作
std::this_thread::sleep_for(std::chrono::seconds(5));

// 析构时自动停止后台线程
```

### 4.8 使用 `tcp_client` 连接到远程日志服务器

```cpp
#include <spdlog/details/tcp_client.h>

spdlog::details::tcp_client client;

// 连接到日志服务器（超时 5000ms）
try {
    client.connect("logserver.example.com", 8080, 5000);

    // 发送日志消息
    std::string log_msg = "{\"level\":\"info\", \"message\":\"Hello from tcp_client\"}\n";
    client.send(log_msg.data(), log_msg.size());

    // 关闭连接
    client.close();
} catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "TCP error: " << ex.what() << "\n";
}
```

### 4.9 使用 `udp_client` 发送日志到远程服务器

```cpp
#include <spdlog/details/udp_client.h>

try {
    // 创建 UDP 客户端（自动连接）
    spdlog::details::udp_client client("192.168.1.100", 514);

    // 发送 syslog 消息
    std::string syslog_msg = "<34>1 2024-01-01T12:00:00Z myapp myproc - - My log message";
    client.send(syslog_msg.data(), syslog_msg.size());

    // 析构时自动关闭套接字
} catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "UDP error: " << ex.what() << "\n";
}
```

### 4.10 使用 `fmt_helper` 格式化输出

```cpp
#include <spdlog/details/fmt_helper.h>
#include <spdlog/common.h>

spdlog::memory_buf_t buf;

// 追加字符串
spdlog::details::fmt_helper::append_string_view("Hello, ", buf);

// 追加整数
spdlog::details::fmt_helper::append_int(42, buf);

// 填充数字（例如时间戳）
spdlog::details::fmt_helper::pad2(5, buf);      // "05"
spdlog::details::fmt_helper::pad3(123, buf);    // "123"
spdlog::details::fmt_helper::pad6(12345, buf);  // "012345"
spdlog::details::fmt_helper::pad9(12345678, buf); // "012345678"

// 计算位数
unsigned int digits = spdlog::details::fmt_helper::count_digits(12345); // 5

// 提取毫秒部分
auto now = spdlog::log_clock::now();
auto millis = spdlog::details::fmt_helper::time_fraction<std::chrono::milliseconds>(now);
spdlog::details::fmt_helper::pad3(millis.count(), buf); // "123"

// 转换为字符串
std::string result(buf.data(), buf.size());
std::cout << result << "\n";
```

## 5. 实现细节

### 5.1 内存管理

- **`circular_q`**: 使用单个 `std::vector<T>` 作为底层存储，预分配 `max_items_ + 1` 个元素（哨兵模式），避免额外的堆分配。移动语义正确处理：移动后源对象进入 disabled 状态（`max_items_ = 0`），防止悬空迭代器。

- **`mpmc_blocking_queue`**: 组合 `circular_q` 和同步原语，队列元素为 `async_msg`（包含 `log_msg_buffer`）。满时根据策略选择阻塞、覆盖或丢弃。

- **`log_msg_buffer`**: 继承自 `log_msg`，内部 `memory_buf_t`（即 `std::vector<char>`）连续存储 `logger_name` 和 `payload`，构造时追加到缓冲区，然后通过 `update_string_views()` 更新 `string_view_t` 指针。避免小对象分配开销。

- **`registry`**: 使用 `std::unordered_map` 存储 logger，`std::shared_ptr` 管理生命周期，`std::unique_ptr` 管理格式化器和周期性刷新器。析构时按顺序清理：先停止 `periodic_flusher_`，再清空 `loggers_`，最后重置 `thread_pool`。

### 5.2 并发控制

- **`mpmc_blocking_queue`**:
  - 使用 `std::mutex` + 双条件变量（`push_cv_`、`pop_cv_`）实现生产者-消费者模式
  - `enqueue`: 持有锁时等待 `pop_cv_` 直到队列非满，入队后通知 `push_cv_`
  - `dequeue`: 持有锁时等待 `push_cv_` 直到队列非空，出队后通知 `pop_cv_`
  - MinGW 特殊处理：锁必须保持到 `notify_one()` 之后（避免死锁）
  - `discard_counter_` 使用 `std::atomic<size_t>` 无锁计数

- **`registry`**:
  - 三把锁保护不同数据：
    - `logger_map_mutex_`: 保护 `loggers_` 和 `default_logger_`
    - `tp_mutex_`: 递归锁，保护线程池（允许嵌套调用）
    - `flusher_mutex_`: 保护 `periodic_flusher_`
  - `get_default_raw()` 返回原始指针（避免 `shared_ptr` 引用计数开销），但要求不并发调用 `set_default_logger()`

- **`backtracer`**:
  - `mutex_` 保护所有队列操作
  - `enabled_` 使用 `std::atomic<bool>` + `memory_order_relaxed`（不需要同步其他变量）

- **`thread_pool`**:
  - 内部 `mpmc_blocking_queue` 已线程安全
  - 工作线程通过 `terminate` 消息优雅退出

- **`null_mutex`**: 零开销优化，单线程场景替代 `std::mutex`，所有方法为空操作

### 5.3 性能优化

- **`circular_q`**:
  - 预分配 `max_items_ + 1` 元素，避免运行时分配
  - 所有操作 O(1)，无锁设计（外部需加锁）
  - `overrun_counter_` 统计覆盖次数，帮助调试队列容量问题

- **`fmt_helper::count_digits_fallback`**:
  - 每 4 次除法处理 4 位数字（相比逐次除 10，减少除法次数）
  - 来自 Alexandrescu "Three Optimization Tips for C++" 演讲技巧

- **`os::thread_id()`**:
  - 使用线程局部缓存（`static thread_local const size_t tid`），避免重复系统调用
  - 比标准 `std::this_thread::get_id()` + `std::hash()` 快得多（尤其 MSVC 2013）

- **`os::now()`**:
  - Linux 支持 `CLOCK_REALTIME_COARSE`（粗粒度时间，减少系统调用开销）

- **`os::fwrite_bytes()`**:
  - 支持 `SPDLOG_FWRITE_UNLOCKED`，使用 `fwrite_unlocked`（Unix）或 `_fwrite_nolock`（Windows），避免内部锁开销

- **`registry::get_default_raw()`**:
  - 返回原始指针而非 `shared_ptr`，避免引用计数原子操作（热路径优化）

- **`log_msg` vs `log_msg_buffer`**:
  - `log_msg` 使用 `string_view_t`（零拷贝，仅引用栈数据）
  - `log_msg_buffer` 仅在需要延长生命周期时使用（如存入队列或回溯器）

### 5.4 错误处理

- **异常类型**: 使用 `spdlog_ex`（继承自 `std::exception`），携带错误消息和 `errno`

- **`file_helper`**:
  - 打开文件失败重试 5 次，间隔 10ms
  - 所有系统调用失败抛 `spdlog_ex`，包含文件名和 `errno`
  - 支持 `file_event_handlers`，允许用户在打开/关闭时注入自定义逻辑

- **`tcp_client` / `udp_client`**:
  - 连接失败尝试所有 `getaddrinfo` 返回的地址（支持 IPv4/IPv6 双栈）
  - Windows 使用 `FormatMessageA` 转换 `WSAGetLastError()` 为可读消息
  - Unix 使用 `gai_strerror` 转换 `getaddrinfo` 错误码

- **`registry`**:
  - 重复注册同名 logger 抛 `spdlog_ex`
  - 初始化失败不影响其他 logger

- **`thread_pool`**:
  - 线程数必须在 1-1000 范围内，否则抛 `spdlog_ex`
  - 析构时捕获所有异常（`SPDLOG_TRY` / `SPDLOG_CATCH_STD`）

### 5.5 依赖关系

- **外部依赖**:
  - `fmt` / `std::format`（格式化库）
  - STL：`<vector>`, `<mutex>`, `<condition_variable>`, `<thread>`, `<atomic>`, `<chrono>`, `<functional>`, `<unordered_map>`, `<tuple>`

- **内部依赖**:
  - `spdlog/common.h`: 公共类型定义（`level`、`source_loc`、`log_clock`、`string_view_t`、`memory_buf_t`、`SPDLOG_INLINE`、`throw_spdlog_ex` 等）
  - `spdlog/logger.h`: `logger` 类定义（`registry` 和 `thread_pool` 需要前向声明）
  - `spdlog/pattern_formatter.h`: `pattern_formatter` 类（`registry` 默认格式化器）

### 5.6 设计模式

- **单例模式**: `registry::instance()` 使用 Meyers' Singleton（函数局部静态变量，C++11 保证线程安全）

- **RAII 模式**:
  - `periodic_worker`: 构造时创建线程，析构时停止并 `join()`
  - `file_helper`: 构造时初始化成员，析构时关闭文件
  - `tcp_client` / `udp_client`: 构造时初始化 Winsock（Windows），析构时清理

- **工厂模式**: `synchronous_factory::create<Sink, ...>()` 模板工厂，根据 sink 类型创建 logger

- **策略模式**: `async_overflow_policy` 枚举（`block`、`overrun_oldest`、`discard_new`），运行时选择队列满时的行为

- **观察者模式**: `file_event_handlers` 回调机制（`before_open`、`after_open`、`before_close`、`after_close`），允许用户观察文件生命周期事件

- **线程局部存储**: `os::thread_id()` 使用 `thread_local` 缓存线程 ID，避免重复系统调用

- **零开销抽象**: `null_mutex` 和 `null_atomic_int`，单线程场景完全避免同步开销

- **桥接模式**: `os` 命名空间桥接不同平台的系统调用差异，统一接口
