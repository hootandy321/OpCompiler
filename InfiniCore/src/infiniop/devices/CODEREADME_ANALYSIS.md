# 目录: devices 架构全景

## 1. 子系统职责

`devices` 目录是 InfiniOP 统一算子框架的**硬件抽象层**，负责将上层算子接口适配到不同的硬件计算平台。该子系统实现了"一处定义，多处运行"的设计理念，为所有 InfiniOP 算子提供跨硬件平台的统一基础设施。

该目录的核心价值在于：
- **硬件解耦**：将算子的数学定义与硬件实现分离，使上层算子无需关心底层硬件差异
- **资源管理**：为每个硬件平台提供设备句柄（Handle）管理、流控制、内存分配等底层运行时支持
- **算子分发**：根据运行时硬件类型，将算子调用路由到正确的硬件后端实现
- **生态兼容**：支持 7 种主流国产和国际硬件平台（NVIDIA、华为昇腾、寒武纪、昆仑芯、摩尔线程、沐曦等）

## 2. 模块导航 (Module Navigation)

### 2.1 通用基础设施

* **pool.h**: 线程安全的无锁对象池
    * 功能: 提供高性能的句柄池管理，用于复用昂贵的硬件资源（如 cuBLAS/cuDNN 句柄）
    * 职责: 实现无锁并发队列，支持多线程环境下的句柄创建、获取和归还
    * 设计要点: 使用 CAS 操作实现无锁栈，避免互斥锁开销

### 2.2 国际硬件后端

* **nvidia/**: NVIDIA GPU 平台支持（CUDA 生态）
    * 功能: 为 NVIDIA GPU（包括 Tesla、A100、H100 等）提供完整的计算支持
    * 职责: 管理 CUDA 流、cuBLAS/cuDNN 库、设备属性查询（warp size、block/grid 限制）
    * 关键文件:
        - `nvidia_handle.{h,cuh}`: NVIDIA 设备句柄，封装 CUDA 运行时
        - `nvidia_common.cu`: 设备属性初始化、cuBLAS/cuDNN 句柄池管理
        - `nvidia_kernel_common.cuh`: CUDA kernel 通用工具（类型转换、索引计算、exp 函数）
    * 支持特性: FP8/BF16/FP16/FP32/FP64、动态 block size 配置（512/1024/4096）
    * 派生支持: 通过继承机制支持 iluvatar（燧原）、qy、hygon 等兼容 CUDA 的设备

* **cpu/**: CPU 通用计算平台
    * 功能: 提供 x86/ARM CPU 的参考实现和调试支持
    * 职责: 实现基于 OpenMP 的并行计算，提供张量索引、padding 等通用工具
    * 关键文件:
        - `cpu_handle.{h,cc}`: CPU 设备句柄（无内部状态，轻量级实现）
        - `common_cpu.{h,cc}`: 索引偏移计算（indexToOffset）、padding 尺寸计算、OpenMP 并行支持
    * 适用场景: 单元测试、性能基准、小规模计算、无 GPU 环境

### 2.3 国产 AI 芯片后端

* **ascend/**: 华为昇腾 NPU 平台（Ascend 910/910B）
    * 功能: 支持华为昇腾 AI 处理器系列（Ascend 910A/910B 等）
    * 职责: 封装 ACL（Ascend Computing Language）和 ACLNN API，管理 NPU 设备和流
    * 关键文件:
        - `ascend_handle.{h,cc}`: 昇腾设备句柄
        - `common_ascend.{h,cc}`: ACL tensor 描述符封装、数据类型转换（infiniDtype_t ↔ aclDataType）
        - `ascend_kernel_common.h`: Ascend C kernel 常量（BLOCK_NUM、BUFFER_NUM、字节对齐）
        - `CMakeLists.txt`: Ascend C kernel 编译配置（SOC_VERSION、CANN_TOOLKIT_HOME）
    * 特色功能:
        - 自定义 tensor 描述符（aclnnTensorDescriptor），支持步长（stride）和存储形状推断
        - 错误消息提取（aclGetRecentErrMsg）用于调试
    * 依赖库: ACL（Ascend Compute Library）、ACLNN（Neural Network API）

* **bang/**: 寒武纪 MLU 平台（Cambricon 云端训练芯片）
    * 功能: 支持寒武纪 MLU 系列（MLU290、MLU370 等）
    * 职责: 封装 CNNL（Cambricon CNN Library）和 CNRT 运行时
    * 关键文件:
        - `bang_handle.{h,cc}`: 寒武纪设备句柄，管理 CNNL 句柄池
        - `common_bang.h`: CNNL tensor 描述符设置、设备拓扑查询（cluster 数、每 cluster 核数）
        - `bang_kernel_common.h`: MLU kernel 工具（索引计算、广播处理、非连续内存拷贝优化）
    * 特色功能:
        - NRAM（Near RAM）优化：计算最优 chunk size，减少 GDRAM ↔ NRAM 搬运
        - 广播输入索引器（InputIndexer），处理多输入广播场景
        - 支持 Cambricon 设备（通过命名空间 cambricon）
    * 硬件特性: NRAM_MAX_SIZE=240KB、ALIGN_SIZE=128 字节
    * 依赖库: CNNL、CNRT

* **kunlun/**: 昆仑芯 XPU 平台（百度昆仑系列）
    * 功能: 支持昆仑芯 XPU 系列（R200、R500 等）
    * 职责: 封装 XPU 运行时和 XDNN（百度 XPU 深度学习库），提供 XBLAS 兼容层
    * 关键文件:
        - `kunlun_handle.{h,cc}`: 昆仑设备句柄，管理 XDNN Context 句柄池
        - `kunlun_common.h`: XPU 类型定义（kunlunStream_t、kunlunEvent_t）、错误码检查宏
        - `kunlun_kernel_common.h`: XPU kernel 工具（自定义 ptrdiff_t/size_t、原子操作、shared memory 操作）
        - `kunlun_xblas.{h,cc}`: cuBLAS 兼容层（允许在 XPU 上使用 CUDA BLAS 接口）
    * 特色功能:
        - 32 位指针扩展到 64 位（padding 机制）便于 DATACOPY 指令
        - 自定义原子操作（支持 half/bfloat16_t）
        - 集成 XDNN 库（百度 xpu::api 命名空间）
    * 依赖库: XPU Runtime、XDNN、XPU Kernel（xtdk）

* **metax/**: 沐曦 MaxX GPU 平台
    * 功能: 支持沐曦 MaxX 系列 GPU
    * 职责: 封装 HCBLAS/HCDNN 库（或 MCBLAS/MCDNN，通过编译选项切换）
    * 关键文件:
        - `metax_handle.{h,cc}`: 沐曦设备句柄，管理 HCBLAS/HCDNN 句柄池
        - `metax_common.h`: 设备属性查询（warp size、block/grid 限制）、数据类型转换
        - `metax_ht2mc.h`: HC → MC API 别名映射（兼容新旧沐曦 SDK）
        - `metax_kernel_common.h`: MaxX kernel 工具（索引计算、exp 函数、FP8/BF16 支持）
    * 特色功能:
        - 双 API 支持：通过 ENABLE_METAX_MC_API 宏切换 HC（旧版）/MC（新版）SDK
        - CUDA 兼容层：类型别名（cuda_bfloat16 → hpcc_bfloat16）便于移植 CUDA 代码
    * 支持特性: FP8/BF16/FP16/FP32、动态 block size（512/1024）
    * 依赖库: HCBLAS/HCDNN（或 MCBLAS/MCDNN）、HC Runtime（或 MC Runtime）

* **moore/**: 摩尔线程 MUSA GPU 平台
    * 功能: 支持摩尔线程 S 系列 GPU（MTT S80、S3000 等）
    * 职责: 封装 MUBLAS 和 MUDNN 库（摩尔线程的 CUDA 兼容实现）
    * 关键文件:
        - `moore_handle.{h,cc}`: 摩尔线程设备句柄，管理 MUBLAS/MUDNN 句柄池
        - `moore_common.h`: 设备属性查询、MUBLAS/MUDNN 错误检查宏
        - `moore_kernel_common.h`: MUSA kernel 工具（索引计算、exp 函数、FP8/BF16 支持）
    * 特色功能:
        - CUDA 兼容层：类型别名（cuda_bfloat16 → mt_bfloat16）
        - exp 函数特殊处理：MUSA SDK 缺少 hexp，手动实现 half/bfloat16 exp
    * 支持特性: FP8/BF16/FP16/FP32、动态 block size（512/1024/2048）
    * 依赖库: MUBLAS、MUDNN、MUSA Runtime

### 2.4 文档状态

所有硬件后端子目录（ascend、bang、cpu、kunlun、metax、moore、nvidia）**均无现有文档**。本分析基于源代码静态分析生成。

## 3. 架构逻辑图解

### 3.1 整体架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                     InfiniOP 上层算子                         │
│                 (src/infiniop/ops/*/)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ 调用 infiniopCreateXXXDescriptor()
                            │ 传入 infiniDevice_t 选择硬件
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              设备句柄分发层 (handle.h)                        │
│   InfiniopHandle { device, device_id }                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  nvidia/     │  │  ascend/     │  │  bang/       │
│  (CUDA)      │  │  (NPU)       │  │  (MLU)       │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │ 继承/别名        │                 │
       ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ iluvatar/    │  │  kunlun/     │  │ cambricon/   │
│ qy/          │  │  (XPU)       │  │              │
│ hygon/       │  │              │  │              │
└──────────────┘  └──────┬───────┘  └──────────────┘
                          │                 │
                          ↓                 ↓
                  ┌──────────────┐  ┌──────────────┐
                  │  metax/      │  │  moore/      │
                  │  (MaxX GPU)  │  │  (MUSA GPU)  │
                  └──────────────┘  └──────────────┘
```

### 3.2 数据流：算子执行流程

以创建并执行一个矩阵乘法算子为例：

1. **算子创建阶段**:
   ```
   infiniopCreateMatmulDescriptor(handle, &desc, ...)
   ├── handle->device == INFINI_DEVICE_NVIDIA
   │   └── 调用 device::nvidia::Handle::create()
   │       └── 初始化 Internal{ _warp_size, _block_size[], ... }
   │
   ├── handle->device == INFINI_DEVICE_ASCEND
   │   └── 调用 device::ascend::Handle::create()
   │       └── 创建 ACL 设备上下文
   │
   ├── handle->device == INFINI_DEVICE_BANG
   │   └── 调用 device::bang::Handle::create()
   │       └── 初始化 Internal{ _cluster_count, cnnl_handles pool }
   ```

2. **算子执行阶段**:
   ```
   infiniopMatmul(desc, workspace, stream, c, a, b)
   ├── 根据 desc->handle->device 路由到硬件实现
   │
   ├── [NVIDIA] 调用 nvidia/cuda/impl.cu
   │   ├── handle->internal()->useCublas(stream, [](cublasHandle_t h) {
   │   │       cublasGemmEx(h, ...);  // 使用 cuBLAS 句柄池
   │   │   });
   │   └── 或调用 CUDA kernel: matmul_kernel<<<grid, block, 0, stream>>>(...);
   │
   ├── [ASCEND] 调用 ascend/impl.cc
   │   ├── aclnnTensorDescriptor a_desc(a), b_desc(b), c_desc(c);
   │   ├── aclnnMatmul(a_desc.tensor, b_desc.tensor, c_desc.tensor, ...);
   │   └── aclrtSynchronizeStream(stream);
   │
   ├── [BANG] 调用 bang/impl.cc
   │   ├── handle->internal()->useCnnl(queue, [](cnnlHandle_t h) {
   │   │       cnnlMatmul(h, ...);  // 使用 CNNL 句柄池
   │   │   });
   │   └── 或调用 BANG kernel: __bang_kernel<<<...>>>(...);
   ```

3. **句柄池管理**（以 NVIDIA 为例）:
   ```
   Pool<cublasHandle_t> blas_handles;
   │
   ├── 第一次请求: pop() → nullopt
   │   └── cublasCreate(&handle) → 新建句柄
   │   └── cublasSetStream(handle, stream) → 绑定流
   │   └── 使用句柄执行计算
   │   └── push(handle) → 归还到池中
   │
   ├── 后续请求: pop() → 返回之前创建的句柄
   │   └── 重复使用，避免创建开销
   ```

### 3.3 关键设计模式

1. **句柄池模式 (Handle Pool)**:
   - 所有 GPU 后端（nvidia、bang、kunlun、metax、moore）均使用 `Pool<T>` 管理 BLAS/DNN 句柄
   - 无锁并发设计，支持多线程同时访问
   - 避免频繁创建/销毁句柄的性能开销（句柄创建通常需要数百毫秒）

2. **类型别名兼容层**:
   ```
   [Moore/Metax] cuda_bfloat16 → mt_bfloat16 / hpcc_bfloat16
   [Kunlun]    cublasHandle_t → kunlun::blas::Handle (XBLAS 兼容层)
   [Metax]     hc* → mc* (通过 ENABLE_METAX_MC_API 宏切换)
   ```
   - 允许直接移植 CUDA 代码到国产 GPU
   - 减少重写工作，降低维护成本

3. **设备继承层次**:
   ```
   InfiniopHandle (基类: device, device_id)
   ├── nvidia::Handle (CUDA 基础实现)
   │   ├── iluvatar::Handle (燧原: 继承 nvidia)
   │   ├── qy::Handle (天数智芯: 继承 nvidia)
   │   └── hygon::Handle (海光: 继承 nvidia)
   │
   ├── bang::Handle (寒武纪基础实现)
   │   └── cambricon::Handle (继承 bang)
   │
   ├── ascend::Handle (独立实现)
   ├── kunlun::Handle (独立实现)
   ├── metax::Handle (独立实现)
   ├── moore::Handle (独立实现)
   └── cpu::Handle (独立实现，无内部状态)
   ```

4. **Kernel 公共工具复用**:
   - 所有后端的 `kernel_common.h` 都提供 `indexToOffset()` 函数
   - 作用：将扁平化索引转换为张量内存偏移量（考虑步长）
   - 用途：处理非连续张量（转置、切片等操作的结果）

### 3.4 性能优化要点

1. **Bang (MLU) 的 NRAM 优化**:
   - GDRAM（全局内存） ↔ NRAM（近内存，类似 CUDA shared memory）搬运是主要瓶颈
   - `calculateChunkSize()` 函数计算最大连续块，减少搬运次数
   - `nonContiguousMemcpy()` 支持非连续张量的分块搬运

2. **NVIDIA/Moore/Metax 的动态 Block Size**:
   - 根据设备架构选择合适的 block size（512/1024/2048/4096）
   - 例如：Hygon DCU 最大支持 1024 线程/block，NVIDIA A100 支持 1024，H100 支持 2048

3. **Kunlun XPU 的 32 位指针优化**:
   - XPU 使用 32 位指针，但为了 DATACOPY 指令对齐，扩展到 64 位
   - 自定义 `_ptrdiff_t` 和 `_size_t` 类型（value + padding）

### 3.5 硬件特性对比表

| 硬件平台 | BLAS 库 | DNN 库 | 编程语言 | 特色功能 |
|---------|---------|--------|---------|---------|
| **NVIDIA** | cuBLAS | cuDNN | CUDA | 生态最成熟，支持 FP8 E4M3 |
| **Ascend** | - | ACLNN | Ascend C | 自定义 tensor 描述符，ACL 错误追踪 |
| **Bang** | - | CNNL | BANG | NRAM 优化，cluster 拓扑查询 |
| **Kunlun** | XBLAS | XDNN | XPU | 自定义原子操作，32 位指针扩展 |
| **Metax** | HCBLAS/MCBLAS | HCDNN/MCDNN | HC/MC | 双 SDK 支持，CUDA 兼容层 |
| **Moore** | MUBLAS | MUDNN | MUSA | CUDA 兼容层，exp 函数特殊处理 |
| **CPU** | - | - | C++ + OpenMP | 参考实现，调试支持 |

## 4. 总结

`devices` 目录是 InfiniOP 的硬件适配核心，通过统一的 `InfiniopHandle` 接口和设备特定的 `Handle` 实现，实现了算子的跨平台部署。其设计亮点包括：

1. **零侵入扩展**：新增硬件平台只需实现 Handle 类和 common/kernel_common 工具，无需修改上层算子代码
2. **性能优先**：句柄池、NRAM 优化、动态 block size 等设计确保高性能
3. **国产化支持**：完整支持 7 种国产/国际 AI 芯片，适配不同 SDK 风格
4. **代码复用**：通过继承和类型别名，最大化复用 CUDA 生态代码

该子系统是 InfiniCore 实现"一次编写，多处运行"愿景的关键基础设施。
