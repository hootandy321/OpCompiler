# 目录: infinirt 硬件运行时抽象层架构全景

## 1. 子系统职责

`infinirt` 是 InfiniCore 的硬件运行时抽象层（Hardware Runtime Abstraction Layer），为上层计算框架提供统一的多硬件后端支持接口。该子系统封装了 7 种国产和国际主流硬件平台的底层 API 差异，包括：

- **设备管理**：统一的设备枚举、选择和同步接口
- **流控制**：跨硬件的异步执行流管理
- **事件同步**：跨平台的任务依赖与性能计时机制
- **内存管理**：屏蔽硬件差异的内存分配、释放和拷贝操作

**设计目标**：实现"一次编写，多硬件运行"的跨平台计算能力，使 InfiniCore 的上层算子实现无需关心底层硬件细节。

## 2. 模块导航

### 2.1 核心调度层

* **📂 infinirt.cc & infinirt_impl.h** (根目录)
  * **功能**：硬件运行时的统一调度入口，基于宏定义的多态分发器
  * **职责**：
    - 维护线程本地设备上下文（`CURRENT_DEVICE_TYPE`、`CURRENT_DEVICE_ID`）
    - 通过 `INFINIRT_CALL_DEVICE_API` 宏实现运行时硬件分发
    - 定义标准化的 21 个硬件运行时 API 函数
  * **设计亮点**：
    - 使用 `thread_local` 存储实现线程级设备隔离
    - 提供设备切换优化（CPU 与非 CPU 设备间的快速切换）

### 2.2 硬件后端实现层

#### 2.2.1 国际主流硬件

* **📂 cuda** (NVIDIA GPU 及兼容生态)
  * **功能**：支持 NVIDIA GPU 及兼容 CUDA API 的国产 GPU（如天数智芯、壁仞、海光等）
  * **职责**：完整的 CUDA Runtime API 适配，包括同步/异步内存分配、事件计时等高级特性
  * **实现文件**：`infinirt_cuda.cu/.cuh`
  * **API 映射**：
    - `cudaStreamCreate` → `infinirtStreamCreate`
    - `cudaMallocAsync` → `infinirtMallocAsync` (支持流有序内存)
    - `cudaEventElapsedTime` → `infinirtEventElapsedTime`
  * **命名空间策略**：根据编译宏自动选择 `cuda`/`iluvatar`/`qy`/`hygon` 命名空间

* **📂 cpu** (CPU 通用处理器)
  * **功能**：基于标准 C++ 库的 CPU 后端实现
  * **职责**：提供最小化兼容实现，用于开发和调试
  * **实现文件**：`infinirt_cpu.cc/.h`
  * **特殊设计**：
    - 设备数量固定为 1
    - 使用 `std::malloc`/`std::memcpy` 实现内存管理
    - 事件基于 `std::chrono` 时间戳实现
    - 流和同步操作为空操作（CPU 同步执行）

#### 2.2.2 国产 AI 芯片后端

* **📂 ascend** (华为昇腾 NPU)
  * **功能**：华为昇腾 AI 处理器的 CANN 运行时适配
  * **职责**：ACL (Ascend Computing Language) API 封装
  * **实现文件**：`infinirt_ascend.cc/.h`
  * **核心特性**：
    - 使用 `aclInit` 进行全局初始化（通过 `std::call_once` 保证线程安全）
    - `aclrtMallocAlign32` 提供 32 字节对齐的设备内存分配
    - 支持 `ACL_STREAM_FAST_LAUNCH` 快速启动模式
  * **API 映射**：
    - `aclrtGetDeviceCount` → `getDeviceCount`
    - `aclrtCreateStreamWithConfig` → `streamCreate`
  * **未实现特性**：
    - `eventCreateWithFlags`：返回 `INFINI_STATUS_NOT_IMPLEMENTED`
    - `eventElapsedTime`：昇腾不支持事件计时

* **📂 bang** (寒武纪 MLU)
  * **功能**：寒武纪思元 MLU 系列的 CNRT 运行时适配
  * **职责**：BangC 开发环境的底层 API 封装
  * **实现文件**：`infinirt_bang.cc/.h`
  * **核心特性**：
    - 使用 `cnrtQueue`/`cnrtNotifier` 实现流和事件
    - `cnrtPlaceNotifier` 记录事件到队列
    - `cnrtQueryNotifier` 查询事件状态（`cnrtErrorBusy` 表示未就绪）
  * **未实现特性**：
    - `eventCreateWithFlags` 和 `eventElapsedTime`：未实现
    - `mallocAsync`/`freeAsync`：回退到同步分配（代码注释明确说明）

* **📂 kunlun** (昆仑 XPU)
  * **功能**：北京昆仑芯 XPU 的 XPU Runtime 适配
  * **职责**：昆仑芯 xpu/runtime.h 封装
  * **实现文件**：`infinirt_kunlun.cc/.h`
  * **核心特性**：
    - 设备同步使用 `xpu_wait()` 等待默认流（注释说明昆仑无设备同步 API）
    - `xpu_memcpy_async` 支持异步拷贝
    - `xpu_host_alloc` 实现主机内存分配（带标志参数）
  * **未实现特性**：
    - `eventCreateWithFlags` 和 `eventElapsedTime`：未实现
    - `eventQuery`：昆仑 2 不支持事件查询（代码注释）
    - `mallocAsync`：昆仑 3 不支持异步内存分配

* **📂 metax** (天数智芯 GPGPU)
  * **功能**：天数智芯通用 GPU 的 HC Runtime 适配
  * **职责**：基于 `hc_runtime.h` 的 API 封装，兼容 MC 系列（MUSA）
  * **实现文件**：`infinirt_metax.cc/.h`
  * **编译选项**：
    - `ENABLE_METAX_MC_API`：启用 MUSA Runtime（`mcr/` 路径）
    - 默认使用 HC Runtime（`hcr/` 路径）
  * **完整特性**：
    - 支持所有事件功能（包括 `eventCreateWithFlags` 和 `eventElapsedTime`）
    - 实现流有序内存管理（`mallocAsync`/`freeAsync`）
    - 支持事件标志映射（`hcEventDisableTiming`/`hcEventBlockingSync`）
  * **API 映射**：
    - `hcMallocAsync` → `mallocAsync`
    - `hcEventCreateWithFlags` → `eventCreateWithFlags`

* **📂 moore** (摩尔线程 GPU)
  * **功能**：摩尔线程 MTT S80 的 MUSA Runtime 适配
  * **职责**：`musa_runtime.h` API 封装
  * **实现文件**：`infinirt_moore.cc/.h`
  * **核心特性**：
    - 使用 `musaStream_t`/`musaEvent_t` 类型定义
    - `musaEventQuery` 支持状态查询（`musaErrorNotReady` 表示未就绪）
    - `musaEventElapsedTime` 支持事件计时
  * **未实现特性**：
    - `eventCreateWithFlags`：返回 `INFINI_STATUS_NOT_IMPLEMENTED`
    - `mallocAsync`/`freeAsync`：回退到同步分配

## 3. 架构逻辑图解

### 3.1 调度分发流程

```
用户调用 infinirt API
         ↓
infinirt.cc 宏分发器 (INFINIRT_CALL_DEVICE_API)
         ↓
┌─────────────────────────────────────────┐
│  线程本地设备上下文 (thread_local)       │
│  - CURRENT_DEVICE_TYPE                  │
│  - CURRENT_DEVICE_ID                    │
└─────────────────────────────────────────┘
         ↓
  根据 device_type 分发到对应命名空间
         ↓
┌──────────┬──────────┬──────────┬──────────┐
│ cuda     │ ascend   │ bang     │ kunlun   │
│ namespace│ namespace│ namespace│ namespace│
├──────────┼──────────┼──────────┼──────────┤
│ metax    │ moore    │ cpu      │          │
│ namespace│ namespace│ namespace│          │
└──────────┴──────────┴──────────┴──────────┘
         ↓
   调用各硬件原生 API
         ↓
    返回统一状态码
```

### 3.2 设备上下文管理机制

**线程本地存储策略**：
```cpp
thread_local infiniDevice_t CURRENT_DEVICE_TYPE = INFINI_DEVICE_CPU;
thread_local int CURRENT_DEVICE_ID = 0;
thread_local infiniDevice_t PREVIOUS_NON_CPU_DEVICE_TYPE;
thread_local int PREVIOUS_NON_CPU_DEVICE_ID;
```

**设备切换优化**：
- 当从 CPU 切换回前一个非 CPU 设备时，跳过硬件 API 调用（`skip_set` 逻辑）
- 避免不必要的 `setDevice` 开销

### 3.3 API 实现矩阵

| 功能类别 | API 函数 | cuda | ascend | bang | kunlun | metax | moore | cpu |
|---------|---------|------|--------|------|--------|-------|-------|-----|
| **设备管理** | getDeviceCount | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 固定1 |
| | setDevice | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 空操作 |
| | deviceSynchronize | ✓ | ✓ | ✓ | xpu_wait | ✓ | ✓ | 空操作 |
| **流控制** | streamCreate | ✓ | FAST_LAUNCH | ✓ | ✓ | ✓ | ✓ | 空操作 |
| | streamDestroy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 空操作 |
| | streamSynchronize | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 空操作 |
| | streamWaitEvent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 未实现 |
| **事件管理** | eventCreate | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 时间戳 |
| | eventCreateWithFlags | ✓ | 未实现 | 未实现 | 未实现 | ✓ | 未实现 | 忽略标志 |
| | eventRecord | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 更新时间戳 |
| | eventQuery | ✓ | ✓ | ✓ | 未实现 | ✓ | ✓ | 立即完成 |
| | eventSynchronize | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 立即返回 |
| | eventDestroy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 释放时间戳 |
| | eventElapsedTime | ✓ | 未实现 | 未实现 | 未实现 | ✓ | ✓ | chrono计算 |
| **内存管理** | mallocDevice | ✓ | 32字节对齐 | ✓ | ✓ | ✓ | ✓ | malloc |
| | mallocHost | ✓ | ✓ | HostMalloc | host_alloc | ✓ | ✓ | 同Device |
| | freeDevice | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | free |
| | freeHost | ✓ | ✓ | ✓ | host_free | ✓ | ✓ | 同Device |
| | memcpy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | memcpy |
| | memcpyAsync | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 同步拷贝 |
| | mallocAsync | ✓ | 同步分配 | 同步分配 | 同步分配 | 同步分配 | ✓ | 同步分配 | 同步分配 |
| | freeAsync | ✓ | 同步释放 | 同步释放 | 同步释放 | 同步释放 | ✓ | 同步释放 | 同步释放 |

### 3.4 数据流交互模式

**跨设备数据传输示例**：
```
CPU 数据准备
    ↓
infinirtMalloc(H2D) → 调用 cudaMalloc / aclrtMallocAlign32 / cnrtMalloc
    ↓
infinirtMemcpy(H2D, stream) → cudaMemcpyAsync / aclrtMemcpyAsync / cnrtMemcpyAsync_V2
    ↓
GPU 计算 (kernel 执行)
    ↓
infinirtMemcpy(D2H, stream) → 数据回传
    ↓
infinirtStreamSynchronize(stream) → 等待完成
```

**事件依赖编排示例**：
```
stream1: Task A → event1.record()
stream2: event1.wait() → Task B → event2.record()
stream3: event2.wait() → Task C
```

### 3.5 编译时硬件选择机制

**条件编译策略**：
- 每个 API 通过宏定义检查硬件支持（`ENABLE_NVIDIA_API`、`ENABLE_ASCEND_API` 等）
- 未启用硬件时提供空操作实现（`INFINIRT_DEVICE_API_NOOP`）
- `infinirt_impl.h` 定义统一接口宏（`INFINIRT_DEVICE_API_IMPL`）

**CUDA 生态多品牌适配**：
```cpp
#if defined(ENABLE_NVIDIA_API)
namespace infinirt::cuda
#elif defined(ENABLE_ILUVATAR_API)
namespace infinirt::iluvatar
#elif defined(ENABLE_QY_API)
namespace infinirt::qy
#elif defined(ENABLE_HYGON_API)
namespace infinirt::hygon
#endif
```

### 3.6 错误处理统一机制

**宏辅助定义**：
```cpp
#define CHECK_INTERNAL(RT_API, SUCCESS_CODE) \
    do { \
        auto _err = RT_API; \
        if (_err != SUCCESS_CODE) { \
            return INFINI_STATUS_INTERNAL_ERROR; \
        } \
    } while(0)
```

**各后端专用检查宏**：
- CUDA: `CHECK_CUDART` (检查 `cudaSuccess`)
- Ascend: `CHECK_ACLRT` (检查 `ACL_SUCCESS`)
- Bang: `CHECK_BANGRT` (检查 `cnrtSuccess`)
- Kunlun: `CHECK_KUNLUNRT` (检查 `XPU_SUCCESS`)
- Metax: `CHECK_MACART` (检查 `hcSuccess`)
- Moore: `CHECK_MUSART` (检查 `musaSuccess`)

### 3.7 性能优化要点

1. **设备切换缓存**：避免重复的硬件 `setDevice` 调用
2. **流有序内存**：CUDA 和 Metax 支持真正的异步内存分配
3. **对齐分配**：Ascend 提供 32 字节对齐优化
4. **零拷贝优化**：CPU 后端跳过不必要的内存拷贝
5. **快速启动模式**：Ascend 使用 `ACL_STREAM_FAST_LAUNCH` 标志

## 4. 关键设计决策

### 4.1 命名空间隔离
每个硬件后端独立命名空间（`infinirt::cuda`、`infinirt::ascend` 等），编译时静态分发，无运行时开销。

### 4.2 线程本地上下文
使用 `thread_local` 实现线程级设备隔离，避免多线程设备切换竞争。

### 4.3 渐进式实现
不同硬件后端可根据实际能力提供实现：
- 完整实现（CUDA、Metax）
- 部分实现 + 回退（Bang、Kunlun 的异步内存）
- 最小实现（CPU）

### 4.4 类型抹除
使用 `void*` 定义 `infinirtStream_t` 和 `infinirtEvent_t`，在分发层强制转换为硬件原生类型。

## 5. 待扩展点

### 5.1 功能缺失
- **Ascend**：事件标志和计时功能未实现
- **Bang/Kunlun/Moore**：异步内存管理未实现
- **Kunlun**：事件查询 API 缺失

### 5.2 新硬件接入流程
1. 创建新目录（如 `/home/qy/src/Infini/InfiniCore/src/infinirt/xxx`）
2. 实现 `INFINIRT_DEVICE_API_IMPL` 要求的 21 个函数
3. 在 `infinirt.cc` 的 `INFINIRT_CALL_DEVICE_API_AND` 宏添加 case
4. 添加编译时开关（`ENABLE_XXX_API`）

## 6. 依赖关系

**向上依赖**：无（为上层提供基础服务）

**向下依赖**：
- 各硬件厂商的官方运行时库（`cuda_runtime.h`、`acl/acl.h`、`cnrt.h` 等）
- InfiniCore 通用工具（`../../utils.h`）
- InfiniCore 基础类型（`infinicore.h`）

**同级依赖**：无（独立硬件抽象层）

---

**文档生成时间**：2026-01-14
**分析范围**：./InfiniCore/src/infinirt（7 个硬件后端 + 核心调度层）
**代码总行数**：约 1500 行（不含注释和空行）
**支持硬件**：NVIDIA、Ascend、Cambricon、Kunlun、Iluvatar、Moore Threads、通用 CPU
