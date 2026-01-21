# 目录: include 头文件层架构全景

## 1. 子系统职责

`include` 目录是 **InfiniCore 框架的公共 API 契约层**，定义了整个系统的对外接口规范和类型系统。该目录作为框架与外部世界交互的唯一入口，承担着以下关键职责：

- **统一类型系统定义**：通过 `infinicore.h` 建立跨模块的类型契约，包括状态码、设备类型枚举、数据类型枚举等基础类型定义
- **硬件抽象接口规范**：通过 `infinirt.h` 定义底层硬件运行时接口，统一设备管理、流控制、事件同步和内存分配的 C 接口
- **算子接口标准**：通过 `infiniop.h` 聚合所有计算算子的头文件，为 LLM 推理提供完整的算子 API 声明
- **分布式通信抽象**：通过 `infiniccl.h` 提供跨设备通信的标准接口，支持张量并行和数据并行的集合通信原语
- **C++ 高层封装**：通过 `infinicore/` 和 `infiniop/` 两个子目录提供 RAII 风格的 C++ API，为开发者提供类型安全和易用性

该目录位于 **InfiniCore 架构的最顶层**，是所有上层应用（InfiniLM、InfiniTrain、InfiniPerf）依赖的接口边界，也是实现多硬件后端可移植性的关键抽象层。

---

## 2. 模块导航

### 2.1 根级头文件（基础设施层）

* **`infinicore.h`**:
    * *功能*: 定义框架的全局类型系统和导出宏，包括状态码枚举、设备类型枚举（10 种硬件）、数据类型枚举（19 种数值类型）
    * *职责*: 作为所有子模块的类型基础，确保跨编译单元的类型一致性，提供跨平台共享库导出机制（Windows `__declspec(dllexport)` / GCC `__attribute__((visibility("default")))`）
    * *关键类型*:
        - `infiniStatus_t`: 统一错误码系统（成功、通用错误、设备错误、张量错误等）
        - `infiniDevice_t`: 硬件后端标识（CPU、NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGO、QY）
        - `infiniDtype_t`: 数据类型标识（BOOL、INT8/16/32/64、UINT8/16/32/64、FP8/16/32/64、BF16、复数类型等）

* **`infinirt.h`**:
    * *功能*: 硬件运行时抽象层接口，提供设备管理、流控制、事件同步、内存分配的 C 函数声明
    * *职责*: 作为硬件驱动层的统一接口契约，屏蔽不同厂商 API 差异（如 CUDA、CANN、BANG 等），为上层提供一致的异步执行模型
    * *接口分类*:
        - **初始化**: `infinirtInit()`
        - **设备管理**: `infinirtGetDeviceCount/SetDevice/GetDevice/Synchronize`
        - **流控制**: `infinirtStreamCreate/Destroy/Synchronize/WaitEvent`
        - **事件同步**: `infinirtEventCreate/Record/Query/Synchronize/ElapsedTime`（支持精确性能测量）
        - **内存管理**: `infinirtMalloc/Free/Memcpy`（支持同步/异步、主机/设备、流有序内存分配）

* **`infiniccl.h`**:
    * *功能*: 集合通信库接口，提供多设备间通信的标准原语
    * *职责*: 支持分布式训练和推理的张量并行，通过 `infinicclComm_t` 句柄抽象通信组，提供 AllReduce 等集合通信操作
    * *接口*:
        - `infinicclCommInitAll`: 初始化跨设备通信组
        - `infinicclAllReduce`: 执行归约操作（SUM/PROD/MAX/MIN/AVG）
        - `infinicclCommDestroy`: 销毁通信句柄

* **`infiniop.h`**:
    * *功能*: 算子库的聚合头文件，统一导出所有算子接口的头文件（handle.h、tensor_descriptor.h 和 33 个算子头文件）
    * *职责*: 作为上层应用的单一包含入口，简化依赖管理，确保所有算子接口的一致性加载
    * *覆盖算子域*: 线性代数、注意力机制、归一化、激活函数、量化、采样、内存操作等 LLM 推理全流程算子

* **`infinicore.hpp`**:
    * *功能*: C++ 兼容性包装（目前内容为空）
    * *职责*: 预留 C++ 命名空间或模板接口扩展点

### 2.2 子目录模块（C++ 封装层）

* **`infinicore/`**: InfiniCore 框架的 C++ 公共 API 头文件层
    * *功能*: 提供从底层硬件抽象到高层神经网络构建的完整 C++ 编程接口体系，包含 6 个子模块（common、context、graph、nn、ops、ops/common）
    * *职责*: 作为框架的"门面"，向下对接 `infinirt` 和 `infiniop` 的 C 接口，向上提供 RAII 风格、类型安全的 C++ API
    * *核心设计*: 通过分层模块化和 `OpDispatcher` 设备分发机制，实现"一次编码，多硬件运行"的目标
    * *详细架构*: 参见 `/home/qy/src/Infini/InfiniCore/include/infinicore/CODEREADME_ANALYSIS.md`

* **`infiniop/`**: InfiniCore 算子接口层的 C++ 头文件封装
    * *功能*: 定义算子接口的 C++ 类型和辅助工具，包含基础设施模块（handle.h、tensor_descriptor.h、operator_descriptor.h）和算子接口模块（ops/ 子目录，33 个算子头文件）
    * *职责*: 提供算子描述符的类型安全定义、统一的生命周期管理接口（Create/GetWorkspace/Execute/Destroy）、张量抽象与描述
    * *核心设计*: 通过不透明描述符类型（`typedef struct InfiniopDescriptor *infiniopXxxDescriptor_t`）隐藏实现细节，实现多态后端支持
    * *详细架构*: 参见 `/home/qy/src/Infini/InfiniCore/include/infiniop/CODEREADME_ANALYSIS.md`

---

## 3. 架构逻辑图解

### 3.1 头文件层次的依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│          根级头文件（类型系统与 C 接口契约）                    │
├─────────────────────────────────────────────────────────────┤
│  infinicore.h      :: 全局类型定义（status/device/dtype）    │
│  infinirt.h        :: 硬件运行时抽象（device/stream/memory） │
│  infiniccl.h       :: 集合通信接口（AllReduce/Comm）         │
│  infiniop.h        :: 算子聚合头文件（33 个算子接口）         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 依赖
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          子目录头文件（C++ 高层封装）                          │
├─────────────────────────────────────────────────────────────┤
│  infinicore/       :: 框架 C++ API（context/graph/nn/ops）  │
│  infiniop/         :: 算子 C++ 类型（descriptor/ops）        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 被包含
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          上层应用（InfiniLM / InfiniTrain / InfiniPerf）     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模块间的接口协作

#### 协作 1: 类型系统统一性

**`infinicore.h` → 所有子模块**

`infinicore.h` 定义的基础类型被所有模块使用：

```
infinicore.h 类型定义
    │
    ├──→ infinirt.h: 使用 infiniStatus_t/infiniDevice_t/infiniDtype_t
    │      （设备管理函数的参数和返回值）
    │
    ├──→ infiniccl.h: 使用 infiniStatus_t/infiniDevice_t/infiniDtype_t
    │      （通信初始化和 AllReduce 的类型参数）
    │
    ├──→ infiniop/ops/*.h: 使用 infiniStatus_t/infiniDtype_t
    │      （算子描述符创建和执行的状态码）
    │
    └──→ infinicore/*.hpp: 使用 infiniDevice_t/infiniDtype_t
           （C++ 封装层的枚举类型映射）
```

**设计优势**：
- 单一数据源：所有模块使用同一套类型定义，避免类型不匹配
- 编译期检查：跨模块函数调用时的类型安全保证
- ABI 稳定性：枚举值的固定定义确保跨版本二进制兼容

#### 协作 2: 硬件抽象分层

**`infinirt.h` → `infinicore/context` → `infinicore/ops` → `infiniop/ops`**

四层抽象体系，从底层硬件驱动到高层算子接口：

```
┌──────────────────────────────────────────────────────────┐
│  应用层调用: ops::add(a, b)  (infinicore/ops)            │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  算子接口层: infiniopAddDescriptor (infiniop/ops/add.h)  │
│   - 定义算子描述符结构                                     │
│   - 声明 Create/GetWorkspace/Execute/Destroy            │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  上下文层: Context (infinicore/context/context.hpp)     │
│   - 封装 infinirtSetDevice/infinirtMalloc               │
│   - 提供 RAII 风格的设备/流/内存管理                      │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  运行时层: infinirt.h (C 接口)                            │
│   - infinirtSetDevice(device, id)                       │
│   - infinirtMalloc(&ptr, size)                          │
│   - infinirtMemcpyAsync(dst, src, size, kind, stream)   │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  硬件驱动层: cuDNN / CANN / BANG / ... (厂商库)          │
└──────────────────────────────────────────────────────────┘
```

**数据流路径**：

1. **设备上下文切换**：
   ```
   ops::add(a, b)
     → context::getDevice() 返回当前设备
     → infinirtGetDevice(&device_type, &device_id)
     → ops/common::OpDispatcher 查找设备特定实现
   ```

2. **内存分配流程**：
   ```
   Tensor::create(shape)
     → context::allocateMemory(size)
     → infinirtMalloc(&ptr, size)
     → cuMemAlloc / hipMalloc / cnrtMalloc (硬件特定)
   ```

3. **异步执行控制**：
   ```
   ops::matmul(a, b, stream)
     → context::getDefaultStream()
     → infinirtStreamCreate(&stream)
     → infiniopMatmul(desc, workspace, size, c, a, b, stream)
     → infinirtStreamSynchronize(stream)
   ```

#### 协作 3: 算子全生命周期管理

**`infiniop.h` 聚合头 → `infiniop/ops/*.h` 算子接口 → `infinicore/ops/*.hpp` C++ 封装**

从 C 接口声明到 C++ 类型安全封装：

```
┌──────────────────────────────────────────────────────────┐
│  infiniop.h (聚合头文件)                                  │
│   #include "infiniop/ops/add.h"                          │
│   #include "infiniop/ops/gemm.h"                         │
│   ... (33 个算子)                                         │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  infiniop/ops/add.h (C 接口定义)                         │
│   typedef struct InfiniopDescriptor *infiniopAddDesc_t; │
│   infiniopCreateAddDescriptor(...)                       │
│   infiniopGetAddWorkspaceSize(...)                      │
│   infiniopAdd(...)                                       │
│   infiniopDestroyAddDescriptor(...)                     │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  infinicore/ops/math.h (C++ 封装)                        │
│   inline Tensor add(const Tensor &a, const Tensor &b)   │
│   {                                                      │
│     // 调用 infiniopAdd                                  │
│     // 处理 infiniStatus_t 错误码                        │
│     // 返回 Tensor 包装的结果                            │
│   }                                                      │
└──────────────────────────────────────────────────────────┘
```

**生命周期标准流程**：

```
1. 创建阶段（编译期优化）
   infiniopCreateGemmDescriptor(handle, &desc, c_desc, a_desc, b_desc)
   └─> 选择最优算法（如 cuBLAS 的 tile 大小）
   └─> 预计算常量（scale、alignment）

2. 查询阶段（内存规划）
   infiniopGetGemmWorkspaceSize(desc, &size)
   └─> 返回临时内存需求
   └─> 上层可预分配全局 workspace 池

3. 执行阶段（运行期计算）
   infiniopGemm(desc, workspace, size, c, a, b, 1.0f, 0.0f, stream)
   └─> 异步执行（通过 stream 参数）
   └─> 支持 batch 并发（不同 stream 并行）

4. 销毁阶段（资源释放）
   infiniopDestroyGemmDescriptor(desc)
   └─> 释放编译期生成的资源（kernel cache）
```

#### 协作 4: 分布式通信协同

**`infiniccl.h` → `infinicore/context` → `infinicore/nn` 张量并行层**

集合通信接口如何支撑分布式训练：

```
┌──────────────────────────────────────────────────────────┐
│  infiniccl.h (C 通信接口)                                 │
│   infinicclCommInitAll(device_type, &comms, ndevice, ...)│
│   infinicclAllReduce(sendbuf, recvbuf, count, dtype, ... │
│                      INFINICCL_SUM, comm, stream)        │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  infinicore/context (通信句柄管理)                        │
│   class Context {                                         │
│     infinicclComm_t comm;  // 通信组句柄                  │
│     void initComm(int world_size, int rank);             │
│     infinicclComm_t getComm() const;                     │
│   };                                                      │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  infinicore/nn/parallel.h (张量并行层)                   │
│   class ColumnParallelLinear {                            │
│     Tensor forward(Tensor input) {                       │
│       // 本地计算                                         │
│       auto local_out = ops::linear(input, weight);       │
│       // AllReduce 聚合分片结果                           │
│       infinicclAllReduce(..., context.getComm(), ...);   │
│       return local_out;                                   │
│     }                                                     │
│   };                                                      │
└──────────────────────────────────────────────────────────┘
```

**通信计算重叠优化**：

```
Stream 0: 计算内核
    ├─> GEMM(input, weight)          [计算密集型]
    └─> infinicclAllReduce(...)       [通信密集型]

Stream 1: 数据传输
    └─> infinirtMemcpyAsync(...)      [PCIe 传输]

通过事件同步协调：
    infinirtEventRecord(event_compute, stream0);
    infinirtStreamWaitEvent(stream1, event_compute);
```

### 3.3 关键设计模式

#### 模式 1: 双层接口设计（C + C++）

**目标**：兼顾性能（C）和易用性（C++）

```
C 接口层（infinirt.h, infiniop.h, infiniccl.h）
    ├── 优势：
    │   • 跨语言兼容性（可被 Python/Go/Rust 调用）
    │   • 稳定的 ABI（共享库版本控制）
    │   • 零开销抽象（直接映射硬件 API）
    └── 劣势：
        • 手动内存管理
        • 无类型安全（void* 指针）
        • 错误处理繁琐（检查返回值）

         │ 通过 C++ 封装层解决

C++ 接口层（infinicore/*.hpp, infiniop/*.hpp）
    ├── 优势：
    │   • RAII 自动资源管理
    │   • 强类型系统（Tensor/Context 类）
    │   • 异常安全（状态码自动转异常）
    │   • STL 兼容（迭代器、算法）
    └── 劣势：
        • 仅限 C++ 调用
        • 可能有额外虚函数开销

实际实现：infinicore/ops/*.hpp 直接调用 infiniop 的 C 接口
```

#### 模式 2: 不透明描述符模式

**目标**：隐藏实现细节，支持多态后端

```c
// infiniop/ops/gemm.h
typedef struct InfiniopDescriptor *infiniopGemmDescriptor_t;

// 后端实现可自由定义结构
struct InfiniopDescriptor {
    infiniDevice_t device;
    union {
        struct {
            cublasHandle_t handle;
            cublasGemmAlgo_t algo;
        } cuda;
        struct {
            void* handle;  // OpenBLAS 或 oneDNN
        } cpu;
    };
};

// 上层代码完全不依赖内部结构
infiniopGemmDescriptor_t desc;
infiniopCreateGemmDescriptor(..., &desc);  // 工厂模式
infiniopGemm(desc, ...);                   // 多态调用
infiniopDestroyGemmDescriptor(desc);
```

#### 模式 3: 统一错误处理

**目标**：跨模块的错误传播链

```
infinirt.h 底层错误
    ├── INFINI_STATUS_DEVICE_NOT_FOUND
    ├── INFINI_STATUS_BAD_PARAM
    └── INFINI_STATUS_INTERNAL_ERROR
             │
             ▼ 被转换为
infiniop.h 算子错误
    ├── INFINI_STATUS_BAD_TENSOR_DTYPE  (不支持的 dtype)
    ├── INFINI_STATUS_BAD_TENSOR_SHAPE  (形状不兼容)
    └── INFINI_STATUS_INSUFFICIENT_WORKSPACE (OOM)
             │
             ▼ 被封装为
infinicore/*.hpp C++ 异常
    ├── std::invalid_argument (参数错误)
    ├── std::runtime_error (运行时错误)
    └── 内部自定义异常类
```

---

## 4. 跨后端可移植性实现

### 4.1 设备枚举驱动分发

`infinicore.h` 定义的 10 种设备类型作为分发键：

```cpp
// infinicore/ops/common/dispatcher.hpp
class OpDispatcher {
    std::array<FunctionPtr, INFINI_DEVICE_TYPE_COUNT> table_;

public:
    void register(infiniDevice_t device, FunctionPtr fn) {
        table_[device] = fn;
    }

    FunctionPtr lookup(infiniDevice_t device) const {
        return table_[device];
    }
};

// 使用示例
dispatcher.register(INFINI_DEVICE_NVIDIA, cuda_add_impl);
dispatcher.register(INFINI_DEVICE_ASCEND, ascend_add_impl);
// ... 10 个后端

// 运行时路由
auto impl = dispatcher.lookup(current_device);
impl(output, input_a, input_b);  // 自动分发到正确实现
```

### 4.2 数据类型统一抽象

`infinicore.h` 定义的 19 种数据类型确保跨硬件语义一致性：

```
FP16 (半精度浮点)
    ├── NVIDIA: __half (2 字节)
    ├── Ascend:  half / float16_t
    ├── CPU:    uint16_t + 软件模拟
    └── 统一接口: INFINI_DTYPE_F16

BF16 (脑浮点)
    ├── NVIDIA: __nv_bfloat16 (2 字节，8 位指数)
    ├── CPU:    uint16_t + 软件模拟
    └── 统一接口: INFINI_DTYPE_BF16

INT8 (量化整数)
    ├── 所有后端: int8_t (1 字节)
    └── 统一接口: INFINI_DTYPE_I8
```

### 4.3 异步执行模型抽象

`infinirt.h` 的流抽象屏蔽不同硬件的异步模型：

```
CUDA:  cudaStream_t (支持 kernel 并发、memcpy overlap)
Ascend: aclrtStream (类似 CUDA Stream)
CPU:   包装为线程池任务队列

统一接口:
    infinirtStreamCreate(&stream);
    infiniopMatmul(..., stream);  // 所有算子接受 stream
    infinirtStreamSynchronize(stream);
```

---

## 5. 扩展指南

### 5.1 添加新硬件后端

1. **扩展设备枚举**：在 `infinicore.h` 中添加 `INFINI_DEVICE_MYHARDWARE = 10`
2. **实现 infinirt 接口**：在 `src/backends/myhardware/` 实现 `infinirt.h` 的所有函数
3. **实现 infiniop 算子**：在 `src/ops/myhardware/` 实现 33 个算子（或按需逐步实现）
4. **注册分发器**：在 `infinicore/ops/common/` 中注册新后端的实现函数
5. **测试验证**：确保 `infiniStatus_t` 错误码正确传播

### 5.2 添加新算子

1. **定义 C 接口**：在 `infiniop/ops/xxx.h` 中声明四阶段 API
2. **实现 C++ 封装**：在 `infinicore/ops/xxx.hpp` 中封装类型安全接口
3. **实现后端代码**：在各硬件目录（`cuda/cpu/ascend/...`）实现算子逻辑
4. **更新聚合头**：在 `infiniop.h` 中添加 `#include "infiniop/ops/xxx.h"`

---

## 6. 相关文档索引

- **全局类型定义**: `/home/qy/src/Infini/InfiniCore/include/infinicore.h`
- **硬件运行时接口**: `/home/qy/src/Infini/InfiniCore/include/infinirt.h`
- **算子接口详细说明**: `/home/qy/src/Infini/InfiniCore/include/infiniop/CODEREADME_ANALYSIS.md`
- **C++ 框架 API**: `/home/qy/src/Infini/InfiniCore/include/infinicore/CODEREADME_ANALYSIS.md`
- **后端实现目录**: `/home/qy/src/Infini/InfiniCore/src/`（具体算子的硬件实现）

---

## 7. 总结

`include` 目录通过 **分层接口设计**、**统一类型系统**、**C/C++ 双层 API**，成功构建了一个 **跨硬件、高性能、易扩展** 的公共接口层。该目录的设计实现了以下目标：

1. **可移植性**：`infinirt.h` 和 `infiniop.h` 的 C 接口确保跨语言调用和稳定 ABI
2. **类型安全**：`infinicore.h` 的统一类型定义避免跨模块类型不匹配
3. **易用性**：`infinicore/` 和 `infiniop/` 的 C++ 封装提供 RAII 和强类型系统
4. **可扩展性**：设备枚举和分发器模式允许无缝添加新硬件后端
5. **性能优化**：不透明描述符和 workspace 抽象支持编译期优化和内存复用

该目录是 InfiniCore 框架与外部世界的 **契约边界**，其设计的合理性直接决定了整个系统的可维护性和可扩展性。
