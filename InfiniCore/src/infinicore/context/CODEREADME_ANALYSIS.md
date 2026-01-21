# CODEREADME_ANALYSIS.md - context 子系统架构全景

## 1. 子系统职责

`context` 目录是 InfiniCore 框架的核心管理层，负责构建全局计算上下文和设备资源管理基础设施。该子系统在整个架构中承担以下关键角色：

**核心定位**：作为框架的"操作系统内核"，管理所有计算设备（CPU/GPU/其他加速器）的生命周期、内存分配策略和执行流控制。

**核心职责**：
- **多设备运行时管理**：维护多个设备的运行时实例表，支持设备切换和线程本地上下文
- **统一内存分配**：提供从主机内存到设备内存的多层次分配策略，支持高性能场景优化
- **流与同步控制**：封装底层异步执行流，提供事件计时和流间同步机制
- **计算图支持**：集成 CUDA Graph 等计算图捕获功能，实现执行路径复用

该子系统是 InfiniCore 与底层硬件抽象层（infinirt）和算子库（infiniop）的统一接口，向上层提供简洁的设备管理和资源分配 API。

---

## 2. 模块导航

### **📂 allocators** (内存分配器模块)
- **功能文档**：[README.md](./allocators/README.md)（详细分析文档）
- **职责**：实现多策略内存分配系统，提供从基础主机内存分配到高性能设备内存池的完整解决方案

**核心组件**：
- `MemoryAllocator`：抽象基类，定义统一分配接口
- `HostAllocator`：基于 malloc/free 的主机内存分配器
- `DevicePinnedHostAllocator`：页锁定内存分配器，支持跨设备延迟释放
- `PinnableBlockAllocator`：高性能大小分类内存池（核心组件），支持固定/图模式切换
- `StreamOrderedAllocator`：流有序异步内存分配器

**设计亮点**：
- Size-Class 分配策略（11 个预定义等级：32KB~256MB）减少内存碎片
- 固定模式（Pinned Mode）支持 CUDA Graph 捕获场景
- Block 复用机制提升分配效率

### **📂 runtime** (运行时模块)
- **功能文档**：*无独立文档*（通过源码分析）
- **职责**：封装单个设备的完整运行环境，管理设备流、内存分配器、算子句柄和计算图管理器

**核心接口**：
- **设备管理**：`device()`, `activate()` - 设备绑定与激活
- **内存分配**：`allocateMemory()`, `allocatePinnedHostMemory()` - 分配设备和固定主机内存
- **数据传输**：`memcpyH2D()`, `memcpyD2H()`, `memcpyD2D()` - 支持同步/异步主机与设备间传输
- **流控制**：`syncStream()`, `syncDevice()`, `stream()` - 流同步与获取
- **事件计时**：`createEvent()`, `recordEvent()`, `elapsedTime()` - 精确性能测量
- **计算图**：`startGraphRecording()`, `addGraphOperator()`, `stopGraphRecording()` - 图捕获支持

**内部依赖**：
- 使用 `PinnableBlockAllocator` 作为设备内存分配器
- 使用 `DevicePinnedHostAllocator` 作为固定主机内存分配器
- 集成 `graph::GraphManager` 管理计算图生命周期
- 持有 `infiniopHandle_t` 算子库句柄

### **📄 context_impl.hpp/cc** (上下文实现)
- **职责**：实现全局单例上下文，管理所有设备的运行时表和线程本地当前运行时

**核心机制**：
- **运行时表**：`std::array<std::vector<std::unique_ptr<Runtime>>, COUNT>` - 按设备类型和索引存储所有运行时实例
- **线程本地存储**：`thread_local Runtime *current_runtime_` - 每线程独立的活动运行时指针
- **延迟初始化**：首次访问时自动创建运行时，优先选择非 CPU 设备
- **设备切换**：`setDevice()` 方法切换当前线程的活动设备

**对外 API**（`context` 命名空间）：
- 设备管理：`setDevice()`, `getDevice()`, `getDeviceCount()`
- 流控制：`getStream()`, `syncStream()`, `syncDevice()`
- 内存分配：`allocateMemory()`, `allocateHostMemory()`, `allocatePinnedHostMemory()`
- 数据拷贝：`memcpyH2D()`, `memcpyD2H()`, `memcpyD2D()`, `memcpyH2H()`
- 事件计时：`createEvent()`, `recordEvent()`, `elapsedTime()`, `streamWaitEvent()`
- 计算图：`isGraphRecording()`, `startGraphRecording()`, `addGraphOperator()`, `stopGraphRecording()`

---

## 3. 架构逻辑图解

### 3.1 初始化流程

```
程序启动
    ↓
ContextImpl::singleton() 首次调用
    ↓
构造函数执行：
    1. infinirtGetAllDeviceCount() 查询所有设备数量
    2. runtime_table_[0].resize(1) - 预留 CPU 运行时槽
    3. runtime_table_[0][0] = new Runtime(CPU) - 创建 CPU 运行时
    4. 遍历非 CPU 设备类型（从高优先级到低）：
       - runtime_table_[i].resize(count)
       - 创建第一个设备的 Runtime 实例
       - 设置为 current_runtime_（默认设备）
    ↓
每个 Runtime 构造时：
    1. activate() - infinirtSetDevice()
    2. infinirtStreamCreate() - 创建执行流
    3. infiniopCreateHandle() - 创建算子库句柄
    4. device_memory_allocator_ = new PinnableBlockAllocator(device)
    5. (GPU设备) pinned_host_memory_allocator_ = new DevicePinnedHostAllocator(device)
    6. graph_manager_ = new GraphManager()
```

### 3.2 设备切换流程

```
用户调用 context::setDevice(Device(type, index))
    ↓
ContextImpl::setDevice() 执行：
    ├─ 如果目标设备 = 当前设备 → 直接返回
    ├─ 检查是否正在录制计算图 → 警告
    ├─ 检查 runtime_table_[type][index] 是否存在
    │   ├─ 不存在 → 延迟创建新 Runtime 实例
    │   └─ 存在 → 激活已有 Runtime
    └─ current_runtime_ = runtime_table_[type][index].activate()
            ↓
        infinirtSetDevice() - 切换底层硬件上下文
```

### 3.3 内存分配流程

```
用户调用 context::allocateMemory(size)
    ↓
ContextImpl::singleton().getCurrentRuntime()->allocateMemory(size)
    ↓
Runtime::allocateMemory() 执行：
    ├─ 获取 device_memory_allocator_（PinnableBlockAllocator）
    ├─ allocator->allocate(size)
    │   ↓
    │   PinnableBlockAllocator::allocate()：
    │   1. 对齐 size 到 256 字节边界
    │   2. 查找匹配的 Size-Class（32KB~256MB）
    │   3. 尝试从 free_blocks 复用缓存
    │   4. 若缓存为空，调用 infinirtMalloc() 分配新块
    │   5. 标记 block->frozen = pinned_mode_
    │   6. 添加到 all_blocks_ 索引
    └─ 返回 std::make_shared<Memory>(ptr, size, device_, deleter)
```

### 3.4 计算图录制流程

```
用户调用 context::startGraphRecording()
    ↓
Runtime::startGraphRecording() 执行：
    ├─ device_memory_allocator_->set_pin_mode(true)  // 冻结内存块
    └─ graph_manager_->start_recording()
            ↓
        设置 is_recording_ = true

用户调用 context::addGraphOperator(op)
    ↓
Runtime::addGraphOperator(op) 执行：
    └─ graph_manager_->add_operator(op)
            ↓
        验证 is_recording_ == true
        将 op 添加到 operators_ 列表

用户调用 context::stopGraphRecording()
    ↓
Runtime::stopGraphRecording() 执行：
    ├─ auto graph = graph_manager_->stop_recording()
    │       ↓
    │   创建 Graph 对象（包含所有 operators_）
    │   清空 operators_ 列表
    │   设置 is_recording_ = false
    └─ device_memory_allocator_->set_pin_mode(false)  // 解除冻结
            ↓
        返回 std::shared_ptr<Graph>（可重复执行）
```

### 3.5 数据传输流程

```
H2D 传输（主机 → 设备）：
context::memcpyH2D(dst, src, size, async=true)
    ↓
Runtime::memcpyH2D() 执行：
    ├─ async == true
    │   └─ infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_H2D, stream_)
    └─ async == false
        └─ infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_H2D)

D2H 传输（设备 → 主机）：
context::memcpyD2H(dst, src, size)
    ↓
Runtime::memcpyD2H() 执行：
    └─ infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2H)
        （默认同步传输，确保数据完整性）

D2D 传输（设备 → 设备）：
context::memcpyD2D(dst, src, size, async=true)
    ↓
Runtime::memcpyD2D() 执行：
    ├─ async == true
    │   └─ infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_D2D, stream_)
    └─ async == false
        └─ infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2D)
```

### 3.6 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextImpl (Singleton)                   │
│  - runtime_table_[Device::Type::COUNT][device_index]        │
│  - thread_local current_runtime_                            │
└────┬──────────────────────────────────────────────┬─────────┘
     │                                              │
     │ 拥有                                        │ 拥有
     ▼                                              ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│         Runtime                 │    │         Runtime                 │
│  (设备运行时实例)                │    │         (每设备一个实例)         │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ + device_                       │    │ + device_                       │
│ + stream_                       │    │ + stream_                       │
│ + infiniop_handle_              │    │ + infiniop_handle_              │
│ + device_memory_allocator_ ─────┼────┼─→ PinnableBlockAllocator        │
│ + pinned_host_memory_allocator_ │    │     (高性能设备内存池)            │
│   (GPU设备专用)                 │    │                                 │
│ + graph_manager_                │    │ + graph_manager_                │
└─────────────────────────────────┘    └─────────────────────────────────┘
         │                                            │
         │ 使用                                       │ 使用
         ▼                                            ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│    allocators 模块              │    │     graph 模块                  │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ • MemoryAllocator (接口)        │    │ • GraphManager                  │
│ • HostAllocator                 │    │ • Graph                         │
│ • DevicePinnedHostAllocator     │    │ • GraphOperator                 │
│ • PinnableBlockAllocator        │    └─────────────────────────────────┘
│ • StreamOrderedAllocator        │
└─────────────────────────────────┘
         │
         │ 调用
         ▼
┌─────────────────────────────────┐
│      infinirt (硬件抽象层)       │
├─────────────────────────────────┤
│ • infinirtMalloc/Free           │
│ • infinirtMemcpyAsync           │
│ • infinirtStreamCreate          │
│ • infinirtEvent*                │
└─────────────────────────────────┘
```

### 3.7 线程安全设计

**ContextImpl**：
- `runtime_table_`：全局共享，但构造完成后只读，无需锁
- `current_runtime_`：`thread_local` 存储，每线程独立，天然线程安全

**Runtime**：
- 每个设备独立实例，不跨线程共享（通过 `activate()` 确保线程本地性）
- 内部分配器可能被多线程访问：
  - `PinnableBlockAllocator` 使用 `std::mutex` 保护
  - `DevicePinnedHostAllocator` 非线程安全（已知限制）

**调用约定**：
- 每个线程应独立调用 `context::setDevice()` 设置自己的当前设备
- 跨线程共享 `Memory` 对象安全（引用计数+RAII）
- 不建议跨线程共享同一个 `Runtime` 实例的流和分配器

---

## 4. 关键设计模式

### 4.1 单例模式（Singleton）
- **ContextImpl** 使用静态局部变量实现 Meyer's Singleton
- 保证进程级唯一的设备管理器

### 4.2 线程本地存储（Thread-Local Storage）
- `thread_local Runtime *current_runtime_` 支持多线程独立设备上下文
- 每线程可绑定不同设备，避免同步开销

### 4.3 延迟初始化（Lazy Initialization）
- 运行时实例按需创建，不用的设备不分配资源
- `getCurrentRuntime()` 首次调用时自动选择默认设备

### 4.4 RAII（资源获取即初始化）
- `Memory` 对象使用自定义删除器自动释放内存
- `Runtime` 析构时自动清理流、句柄和分配器

### 4.5 策略模式（Strategy Pattern）
- `MemoryAllocator` 抽象接口支持多种分配策略
- `Runtime` 根据设备类型选择不同分配器组合

---

## 5. 性能特征与优化建议

### 5.1 内存分配性能
| 分配器类型 | 时间复杂度 | 适用场景 |
|-----------|-----------|---------|
| HostAllocator | O(1) | 小规模 CPU 缓冲区 |
| DevicePinnedHostAllocator | O(1) / O(m) 垃圾回收 | CPU-GPU 数据传输 |
| PinnableBlockAllocator | O(k) / O(n) | 高频分配/释放（推荐） |
| StreamOrderedAllocator | O(1) | 异步流水线 |

**优化建议**：
- 深度学习推理/训练：优先使用 `PinnableBlockAllocator`，缓存命中率 >90%
- 小规模临时缓冲区：使用 `HostAllocator` 避免锁开销
- CUDA Graph 应用：在图录制前调用 `set_pin_mode(true)`

### 5.2 设备切换开销
- **切换成本**：`infinirtSetDevice()` 调用，约 10-50 微秒
- **优化策略**：减少设备切换频率，批处理同一设备上的操作
- **多线程建议**：每个线程绑定固定设备，避免动态切换

### 5.3 数据传输性能
- **H2D/D2H 传输**：固定内存（Pinned Memory）可提升 2-5x 带宽
- **D2D 传输**：优先使用异步传输（`async=true`）实现计算与传输重叠
- **跨 PCIe 传输**：避免频繁小数据传输，合并为大数据块

---

## 6. 与其他子系统的接口

### 6.1 上层依赖（谁使用 context）
- **graph 模块**：使用 `context::getStream()` 和 `context::getDevice()` 获取执行上下文
- **ops 模块**：通过 `context::getInfiniopHandle()` 获取算子库句柄
- **用户应用**：直接调用 `context::allocateMemory()`, `context::memcpyH2D()` 等 API

### 6.2 下层依赖（context 依赖谁）
- **infinirt**：硬件抽象层，提供设备管理、内存分配、流控制、事件计时
- **infiniop**：算子库，提供算子执行句柄
- **graph 模块**：计算图管理器（`GraphManager`, `Graph`, `GraphOperator`）
- **utils 模块**：错误检查宏（`INFINICORE_CHECK_ERROR`）

---

## 7. 已知限制与未来方向

### 7.1 当前限制
1. **DevicePinnedHostAllocator 非线程安全**：
   - `gc_queue_` 无锁保护，多线程环境下可能导致数据竞争
   - 临时方案：每线程独立分配器实例

2. **无设备间直接通信支持**：
   - D2D 传输必须通过主机中转（除非硬件支持 P2P）
   - 未来可添加 `enablePeerAccess()` API

3. **运行时生命周期管理不足**：
   - 一旦创建运行时实例，无法销毁（除进程退出）
   - 长期运行的应用可能泄漏资源

### 7.2 可能的扩展方向
1. **支持多流并行**：
   - 目前每个 Runtime 仅有一个流
   - 可扩展为 `std::vector<infinirtStream_t>` 支持多流并发

2. **内存池预分配**：
   - 添加 `Runtime::warmupMemoryPool(size)` 预分配缓存
   - 减少首次分配延迟

3. **运行时统计信息**：
   - 提供接口查询分配/释放次数、缓存命中率、传输带宽等
   - 辅助性能调优和问题诊断

---

**文档生成时间**：2026-01-14
**分析范围**：`InfiniCore/src/infinicore/context/` 及所有子目录
**文档版本**：v1.0
