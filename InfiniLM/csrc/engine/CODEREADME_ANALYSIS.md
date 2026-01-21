# Engine 模块架构全景

## 1. 子系统职责

Engine 模块是 InfiniLM 推理系统的核心执行引擎，负责协调分布式推理工作流。该模块实现了**多级并行架构**，在上层提供统一的推理接口，底层通过多线程工作器和设备间通信实现高性能张量并行推理。

**核心职责域**：
- **分布式编排管理**：基于张量并行（Tensor Parallelism）的多设备协作
- **异步任务调度**：多线程 Worker 模式实现并发推理执行
- **模型生命周期管理**：参数加载、前向传播、KV 缓存管理
- **设备通信抽象**：封装 InfiniCCL 通信库，提供跨设备集合通信能力

**在 InfiniLM 架构中的定位**：
- **上游依赖**：接收来自应用层的推理请求
- **下游调度**：驱动 Models（模型计算）和 Cache（KV 缓存）模块
- **横向协作**：通过 InfiniCCL 实现多设备间的张量分片同步

---

## 2. 模块导航

### 📂 **distributed/**
- **功能**：分布式训练引擎通信管理模块，负责张量并行（Tensor Parallelism）环境下的设备通信组初始化、配置管理和资源生命周期维护
- **职责**：提供多设备间的高性能通信抽象，管理 InfiniCCL 通信域的创建与销毁，封装设备拓扑配置（rank 到物理设备 ID 的映射关系）

### 📄 **infer_engine.hpp/cpp** (当前目录)
- **功能**：推理引擎的总控制器，聚合多个 RankWorker 实例和通信组，提供单卡到多卡并行的统一接口
- **职责**：管理分布式配置、协调多 worker 参数加载、执行并行前向传播、全局 KV 缓存重置

### 📄 **rank_worker.hpp/cpp** (当前目录)
- **功能**：单 rank 独立执行单元，封装模型实例、KV 缓存和异步任务处理逻辑
- **职责**：运行独立线程处理参数加载、前向推理、缓存重置等任务，通过互斥锁和条件变量实现线程安全通信

---

## 3. 架构逻辑图解

### 3.1 整体数据流

```
应用层推理请求
     ↓
InferEngine (总控制器)
     ├─→ [多路分发] → RankWorker #0 (设备 0)
     ├─→ [多路分发] → RankWorker #1 (设备 1)
     └─→ ...        → RankWorker #N (设备 N)
          ↓
     [张量分片计算]
          ↓
     [AllReduce/AllGather] ← CommunicationGroup (InfiniCCL)
          ↓
     各 Rank 输出聚合 → InferEngine → 应用层
```

### 3.2 模块交互序列

#### **初始化阶段** (Startup Phase)
1. **配置解析**：
   - 用户构建 `DistConfig` 指定张量并行拓扑（如 4 卡映射到设备 ID [0,2,4,6]）
2. **通信组创建**：
   - `InferEngine` 构造 `CommunicationGroup`，自动调用 `infinicclCommInitAll` 批量初始化通信域
3. **Worker 派生**：
   - 为每个 rank 创建独立 `RankWorker` 线程，各绑定对应物理设备上下文
   - 每个 Worker 初始化专属的 `InfinilmModel` 实例和 `Cache` 实例

#### **参数加载阶段** (Parameter Loading)
1. **全局分发**：
   - `InferEngine::load_param(name, param)` 接收完整张量参数
2. **局部提取**：
   - 各 `RankWorker::load_param()` 根据本地 rank ID 提取张量分片（如 attention weight 的列分片）
3. **同步等待**：
   - 主线程等待所有 Worker 完成加载（通过 `wait()` 方法阻塞）

#### **推理执行阶段** (Inference Execution)
1. **任务提交**：
   - `InferEngine::forward(input)` 将输入数据广播到所有 Worker
   - 各 Worker 的 `run()` 方法通过条件变量唤醒工作线程
2. **并行计算**：
   - 每个 RankWorker 在独立线程中执行 `model_->forward()`
   - 模型层内部调用 `infinicclAllReduce/AllGather` 同步中间结果（由 `CommunicationGroup` 提供的 `comm` 句柄）
3. **结果收集**：
   - 主线程调用 `worker->wait()` 等待所有 rank 完成
   - 各 Worker 通过 `get_output()` 返回本地生成的 token ID
   - `InferEngine` 聚合输出（通常仅 rank 0 的结果有效，或需进一步后处理）

#### **缓存管理阶段** (Cache Management)
1. **动态重置**：
   - `InferEngine::reset_cache(new_config)` 触发全局缓存重建
   - 各 Worker 销毁旧 Cache 实例，根据新配置分配 GPU 内存
2. **Paged Attention 支持**：
   - 输入数据包含 `block_tables` 和 `slot_mapping` 张量，实现连续批量的分页 KV 缓存

### 3.3 关键依赖关系

#### **InferEngine 依赖链**：
```
InferEngine
 ├─→ distributed::CommunicationGroup (设备通信管理)
 └─→ RankWorker[] (执行单元)
      ├─→ InfinilmModel (计算核心)
      └─→ cache::Cache (KV 存储)
```

#### **CommunicationGroup 生命周期**：
- **创建时机**：`InferEngine` 构造时，先于所有 Worker
- **销毁时机**：`InferEngine` 析构时，晚于所有 Worker（确保通信域在模型释放前仍有效）
- **资源清理**：RAII 模式，析构函数自动调用 `infinicclCommDestroy`

#### **线程同步机制**：
- **生产者-消费者模式**：
  - 主线程提交任务（设置 `has_job_=true`，通知 `cv_`）
  - Worker 线程唤醒执行（检查 `job_cmd_`，完成后设置 `job_done_=true`）
  - 主线程阻塞等待（`wait()` 方法循环检查 `job_done_`）

### 3.4 设计模式与原则

#### **层次化抽象**：
1. **通信层** (`distributed`)：与具体模型解耦，专注设备间通信原语
2. **执行层** (`RankWorker`)：封装单设备完整推理流程，独立线程隔离
3. **编排层** (`InferEngine`)：提供高层接口，屏蔽分布式细节

#### **资源管理模式**：
- **RAII (Resource Acquisition Is Initialization)**：
  - `CommunicationGroup` 管理通信域生命周期
  - `RankWorker` 线程在析构时自动 join
- **零堆分配设计**：
  - 依赖 STL 容器（`std::vector`, `std::unique_ptr`）自动管理内存
  - 避免裸指针和手动 `new/delete`

#### **并发控制策略**：
- **线程安全**：
  - `RankWorker` 内部使用 `std::mutex` + `std::condition_variable` 保护共享状态
  - `get_output()` 返回值副本，避免数据竞争
- **无锁通信**：
  - `CommunicationGroup` 假设外部同步，内部无锁（InfiniCCL 句柄线程安全）

---

## 4. 扩展性分析

### 4.1 硬件后端扩展
- 当前设计支持 `infinicore::Device::Type::CUDA/ROCm/BANG` 等
- 扩展新硬件仅需：
  1. 在 `DistConfig` 中配置新设备 ID
  2. 确保 InfiniCCL 支持该硬件的通信原语

### 4.2 并行策略扩展
- 当前仅实现张量并行（TP），代码结构预留扩展点：
  - 可添加流水线并行（Pipeline Parallelism）的 Worker 分组
  - 可扩展数据并行（Data Parallelism）的 AllReduce 聚合逻辑

### 4.3 缓存策略扩展
- 支持 PagedAttention 和传统连续缓存
- 通过 `CacheConfig` 多态切换不同缓存实现
- 可扩展至 Multi-Query Attention、Grouped-Query Attention 等变体

---

## 5. 性能特征

### 5.1 并行度
- **设备级并行**：张量并行规模 `tp_size`（如 4/8 卡）
- **线程级并行**：每个 RankWorker 独立线程，避免阻塞主线程
- **流水线隐藏**：参数加载与前向计算可并发（多线程）

### 5.2 通信开销
- **批量初始化**：`infinicclCommInitAll` 一次性创建所有通信域，减少 setup 时间
- **按需通信**：仅 `tp_size > 1` 时创建通信资源，单卡场景零开销
- **拓扑优化**：`DistConfig` 支持非连续设备映射，可优化 PCIe 拓扑（如使用不同 NUMA 节点）

### 5.3 内存占用
- **模型参数**：各 rank 分片存储（如 4 卡 TP，每卡仅存 1/4 权重）
- **KV 缓存**：独立管理，可按需重置配置
- **通信缓冲区**：由 InfiniCCL 内部管理，对用户透明

---

## 6. 异常处理与容错

### 6.1 快速失败策略
- 所有 InfiniCCL 调用通过 `RUN_INFINI` 宏包装，错误时抛出异常
- 构造函数失败时立即终止，避免部分初始化状态

### 6.2 资源泄漏防护
- RAII 模式确保通信域和线程在任何异常路径下正确释放
- 析构函数检查指针有效性（如 `tp_size > 1` 时才销毁通信域）

### 6.3 线程安全保证
- `RankWorker` 的公共方法（`load_param`, `run`, `wait`）均加锁保护
- 输出数据通过值传递，避免引用外泄导致数据竞争

---

## 7. 使用场景示例

### 场景 1：单卡推理
```cpp
// 默认配置自动退化为单设备模式
InferEngine engine(model_config);
engine.forward(input);  // 内部仅创建一个 RankWorker，无通信开销
```

### 场景 2：4 卡张量并行
```cpp
// 指定使用设备 0,2,4,6（避免单卡过热）
DistConfig config({0, 2, 4, 6});
InferEngine engine(model_config, config, Device::Type::CUDA);
engine.load_param("layer.weight", tensor);  // 自动分片到 4 个 rank
auto output = engine.forward(input);        // 并行计算 + AllReduce
```

### 场景 3：动态切换缓存策略
```cpp
// 初始使用连续缓存
CacheConfig continuous_cache(/*...*/);
InferEngine engine(model_config, config, Device::CUDA, &continuous_cache);

// 运行时切换到分页缓存（如因长序列需求）
CacheConfig paged_cache(/* page_size=16, ... */);
engine.reset_cache(&paged_cache);  // 所有 rank 同步重建 KV Cache
```

---

## 8. 总结

Engine 模块通过**层次化抽象**和**异步编排**，成功将分布式推理的复杂性封装在简洁的 API 后。其核心价值在于：

1. **统一的单卡-多卡接口**：用户无需修改代码即可从单卡扩展到多卡
2. **高效的资源管理**：RAII + 多线程 + 通信优化，实现高吞吐低延迟
3. **灵活的配置能力**：支持自定义设备拓扑、动态缓存策略
4. **可扩展架构**：为未来混合并行、新硬件后端预留清晰扩展点

该模块是 InfiniLM 实现生产级推理服务的关键基础设施，充分体现了高性能系统设计中"**简单性、可组合性、可预测性**"的工程原则。
