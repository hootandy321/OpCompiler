# 分页缓存操作 (Paged Caching) 架构全景

## 1. 子系统职责

本模块实现了**分页缓存操作（Paged Caching）**，这是大模型推理中 KV Cache 管理的关键优化技术。该操作负责将输入的 Key-Value 对写入分页管理的缓存块中，支持 F16/BF16/F32 数据类型。该子系统采用分层架构设计，将通用接口定义、后端实现和内核实现解耦，支持多硬件后端扩展。

在 InfiniOP 整体架构中，本模块属于核心算子层，为上层推理框架提供高效的 KV Cache 写入能力，是实现 PagedAttention、vLLM 等先进推理优化技术的基础组件。

## 2. 模块导航

* **📂 cuda**:
  * *功能*: CUDA 后端的内核实现定义，包含分页缓存的核心 CUDA 内核函数模板
  * *职责*: 提供跨 NVIDIA GPU 架构的通用内核实现，支持 F16/BF16/F32 数据类型，实现线程块级别的并行写入逻辑
  * *状态*: 文档缺失（仅有内核头文件 `kernel.cuh`）

* **📂 nvidia**:
  * *功能*: NVIDIA GPU 后端的完整实现，封装设备特定的描述符类和内核调度逻辑
  * *职责*: 实现 NVIDIA 特定的分页缓存操作符，提供设备能力查询、内核配置选择、运行时执行等完整功能链路
  * *核心组件*: `Descriptor` 类（封装操作符）、`Opaque` 结构体（隐藏设备细节）、`PagedCachingInfo`（元数据验证）

## 3. 架构逻辑图解

### 3.1 数据流向

```
用户层（推理框架）
    ↓
创建描述符（Descriptor::create）
    ├→ 验证张量描述符（PagedCachingInfo::create）
    └→ 初始化设备句柄（Opaque::internal）
    ↓
执行分页缓存写入（Descriptor::calculate）
    ├→ 查询设备能力（maxThreadsPerBlock）
    ├→ 选择最优内核配置（1024/512/4096 线程）
    └→ 启动 CUDA 内核（launchKernel）
            ↓
    ┌───────────────────────┐
    │  cuda/kernel.cuh      │ ← 内核实现（通用 CUDA）
    │  pagedCaching<T>()    │
    └───────────────────────┘
            ↓
    GPU 设备执行并行写入
    （分页管理的 K/V Cache）
```

### 3.2 模块交互关系

**nvidia → cuda**:
- NVIDIA 后端依赖 CUDA 通用内核实现（`#include ../cuda/kernel.cuh`）
- NVIDIA 后端负责设备特定的调度逻辑（选择线程数、配置网格维度）
- CUDA 后端提供纯计算的内核模板（数据并行写入逻辑）

**nvidia → 上层接口**:
- 通过 `paged_caching.h` 暴露统一的 C 接口（`infiniopCreatePagedCachingDescriptor` 等）
- 通过 `operator.cc` 注册到全局算子工厂，支持运行时动态创建

### 3.3 设计模式应用

**分层架构**:
- **接口层** (`paged_caching.h`, `operator.cc`): 定义统一的 C API，屏蔽硬件差异
- **后端层** (`nvidia/`): 实现硬件特定的描述符和调度逻辑
- **内核层** (`cuda/kernel.cuh`): 实现通用的并行计算内核

**PIMPL 模式**:
- `Descriptor` 类通过 `Opaque` 结构体隐藏 CUDA 类型定义
- 避免头文件污染，支持在不暴露 GPU 实现细节的情况下编译上层代码

**策略模式**:
- 运行时根据 GPU 架构能力（`maxThreadsPerBlock`）选择不同的内核配置
- 支持同一套代码适配不同 GPU（A100/H100/V100/RTX 4090）

**工厂方法**:
- `Descriptor::create()` 封装对象创建、验证、初始化全流程
- 返回统一的状态码（`infiniStatus_t`），简化错误处理

### 3.4 关键技术细节

**内存管理**:
- 描述符生命周期：堆分配（`new`），手动释放（`delete`）
- 设备句柄：共享指针（`std::shared_ptr`）自动管理
- GPU 缓冲区：由调用方管理（k_cache, v_cache, k, v, slot_mapping）

**并发执行**:
- 基于 CUDA 流（`cudaStream_t`）的异步执行
- 网格维度：`(num_kv_heads, num_tokens, 1)`，每个 token-head 对一个线程块
- 块维度：根据设备能力动态选择（1024/512/4096 线程）

**性能优化**:
- 模板特化：为不同数据类型（F16/BF16/F32）生成专用内核
- 零拷贝：内核不使用共享内存（`shared_mem_size = 0`），减少启动开销
- 自适应调度：运行时查询设备能力，选择最优线程配置

**错误处理**:
- 编译期验证：张量维度、数据类型、步长一致性检查
- 运行时检查：设备架构兼容性验证（`maxThreadsPerBlock >= 512`）
- 状态码返回：使用 `Result<T, E>` 模式和 `infiniStatus_t` 枚举

### 3.5 扩展点

本架构设计支持新增其他硬件后端（如 AMD ROCm、华为 Ascend），只需：
1. 创建新的子目录（如 `rocm/`, `ascend/`）
2. 实现对应的 `Descriptor` 类和 `Opaque` 结构体
3. 调用通用的内核实现或提供硬件特定内核
4. 在 `operator.cc` 中注册新的后端工厂函数

这种设计确保了代码的可维护性和可扩展性，同时保持了各硬件后端的独立性。
