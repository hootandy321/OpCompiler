# 目录: infiniccl 架构全景

## 1. 子系统职责

infiniccl 是 Infini 框架的**集合通信抽象层**，负责为分布式训练提供跨硬件平台的统一通信接口。该子系统位于 InfiniCore 的核心运行时层，通过适配器模式封装了多种硬件厂商的集合通信库（NCCL、HCCL、CNCL、BKCL、MCCL 等），为上层提供与具体硬件无关的分布式通信原语。

**核心价值**：
- **硬件无关性**：允许上层训练框架代码无需修改即可运行在 NVIDIA GPU、华为昇腾、寒武纪、昆仑、摩尔线程、沐曦等多种加速卡上
- **统一接口**：标准化 AllReduce 等集合通信操作，屏蔽底层通信库的 API 差异
- **编译时分派**：通过预处理器宏和命名空间隔离，实现零运行时开销的后端选择

该子系统是 Infini 实现"一次编写，多处部署"跨平台能力的关键基础设施之一。

---

## 2. 模块导航 (Module Navigation)

### 2.1 核心调度层

* **📄 infiniccl.cc / infiniccl_impl.h**
    * **功能**：统一通信接口的分发器，基于设备类型路由到具体后端实现
    * **职责**：实现 `infinicclCommInitAll`、`infinicclCommDestroy`、`infinicclAllReduce` 三个公开 API，通过 switch-case 语句根据 `device_type` 将调用转发到对应命名空间的实现
    * **设计亮点**：采用宏定义辅助代码生成（如 `COMM_INIT_ALL(CASE_, NAMESPACE_)`），减少重复代码并提升可维护性

* **📄 infiniccl_impl.h**
    * **功能**：定义后端适配器的通用接口模板
    * **职责**：
        - 定义 `InfinicclComm` 结构体（包含 `device_type`、`device_id`、`comm` 三元组）
        - 提供 `INFINICCL_DEVICE_API` 宏模板，强制所有后端实现相同的三个核心函数
        - 支持 NOOP 实现（当特定后端未编译时返回 `DEVICE_TYPE_NOT_SUPPORTED` 错误）

---

### 2.2 硬件后端实现（7个）

#### **📂 cuda/**（NVIDIA/天数智芯/青云/海光）
* **底层通信库**：NCCL (NVIDIA Collective Communications Library)
* **支持设备**：NVIDIA GPU (INFINI_DEVICE_NVIDIA)、天数智芯 (ILUVATAR)、青云 (QY)、海光 (HYGON) —— 四者共用 NCCL 接口
* **关键实现**：
    - 文件：`infiniccl_cuda.cu`（使用 CUDA 编译）
    - 数据类型映射：`ncclFloat`, `ncclHalf`, `ncclBfloat16`
    - 归约操作映射：`ncclSum`, `ncclProd`, `ncclMax`, `ncclMin`, `ncclAvg`
    - Stream 转换：`getCudaStream()` 将 `infinirtStream_t` 转为 `cudaStream_t`
* **编译条件**：需定义 `ENABLE_NVIDIA_API`/`ENABLE_ILUVATAR_API`/`ENABLE_QY_API`/`ENABLE_HYGON_API` 且非 Windows 系统

#### **📂 ascend/**（华为昇腾）
* **底层通信库**：HCCL (Huawei Collective Communication Library)
* **支持设备**：华为昇腾 AI 处理器 (INFINI_DEVICE_ASCEND)
* **关键实现**：
    - 文件：`infiniccl_ascend.cc`
    - 特殊初始化流程：调用 `HcclCommInitAll` 前需预先对所有设备调用 `aclrtSetDevice`（逆向遍历 device_ids）
    - 数据类型映射：`HCCL_DATA_TYPE_FP32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_BFP16`
    - 归约操作映射：`HCCL_REDUCE_SUM`, `HCCL_REDUCE_PROD`, `HCCL_REDUCE_MAX`, `HCCL_REDUCE_MIN`
    - Stream 转换：`getAscendStream()` 转为 `aclrtStream`
* **编译条件**：需定义 `ENABLE_ASCEND_API`

#### **📂 kunlun/**（昆仑）
* **底层通信库**：BKCL (Biren Kunlun Collective Library)
* **支持设备**：昆仑芯片 (INFINI_DEVICE_KUNLUN)
* **关键实现**：
    - 文件：`infiniccl_kunlun.cc`
    - 数据类型映射：`BKCL_FLOAT`, `BKCL_FLOAT16`, `BKCL_BFLOAT16`
    - 归约操作映射：`BKCL_ADD`（而非 SUM）, `BKCL_PRODUCT`, `BKCL_MAX`, `BKCL_MIN`
    - Stream 转换：`getKunlunStream()` 转为 `XPUStream`（使用 reinterpret_cast）
    - 通信器销毁：使用 `bkcl_destroy_context`（而非通用的 Destroy）
* **编译条件**：需定义 `ENABLE_KUNLUN_API`

#### **📂 cambricon/**（寒武纪）
* **底层通信库**：CNCL (Cambricon Network Collective Library)
* **支持设备**：寒武纪 MLU (INFINI_DEVICE_CAMBRICON)
* **关键实现**：
    - 文件：`infiniccl_cambricon.cc`
    - 复杂初始化流程：
        1. 构造 rank_list（0 到 ndevice-1）
        2. 对每个 device 调用 `cnrtSetDevice`
        3. 调用 `cnclInitComms`（需传入 rank_list 和 ndevice 参数）
    - 数据类型映射：`cnclFloat32`, `cnclFloat16`, `cnclBfloat16`
    - Stream 转换：`getCambriconStream()` 转为 `cnrtQueue_t`
    - 通信器销毁：使用 `cnclFreeComm`
* **编译条件**：需定义 `ENABLE_CAMBRICON_API`

#### **📂 metax/**（沐曦）
* **底层通信库**：HCCL（沐曦兼容华为接口）/ MCCL（沐曦自己的库）
* **支持设备**：沐曦 GPU (INFINI_DEVICE_METAX)
* **关键实现**：
    - 文件：`infiniccl_metax.cc`
    - 双模式支持：
        - `ENABLE_METAX_MC_API` 定义时使用 MCCL + `mcr/mc_runtime_api.h`
        - 否则使用 HCCL + `hcr/hc_runtime_api.h`
    - 数据类型映射：`hcclFloat`, `hcclHalf`, `hcclBfloat16`
    - 归约操作映射：`hcclSum`, `hcclProd`, `hcclMax`, `hcclMin`, `hcclAvg`
    - Stream 转换：`getMacaStream()` 转为 `hcStream_t`
* **编译条件**：需定义 `ENABLE_METAX_API`

#### **📂 moore/**（摩尔线程）
* **底层通信库**：MCCL (Moore Collective Communication Library)
* **支持设备**：摩尔线程 MUSA (INFINI_DEVICE_MOORE)
* **关键实现**：
    - 文件：`infiniccl_moore.cc`
    - 数据类型限制：仅支持 `INFINI_DTYPE_F32` 和 `INFINI_DTYPE_F16`（不支持 BF16）
    - 数据类型映射：`mcclFloat`, `mcclHalf`
    - 归约操作映射：`mcclSum`, `mcclProd`, `mcclMax`, `mcclMin`, `mcclAvg`
    - Stream 转换：`getMusaStream()` 转为 `musaStream_t`
    - 运行时库：使用 `<musa_runtime.h>`
* **编译条件**：需定义 `ENABLE_MOORE_API`

#### **📂 [公共头]** include/infiniccl.h
* **功能**：定义对外的 C API 接口
* **职责**：
    - 定义归约操作枚举：`INFINICCL_SUM`/`PROD`/`MAX`/`MIN`/`AVG`
    - 声明不透明类型 `struct InfinicclComm` 及其指针类型 `infinicclComm_t`
    - 声明三个导出函数：`infinicclCommInitAll`, `infinicclCommDestroy`, `infinicclAllReduce`

---

## 3. 架构逻辑图解

### 3.1 调用链路（自顶向下）

```
用户代码（InfiniTrain/InfiniLM）
    ↓ 调用 C API
infiniccl.cc (分发器层)
    ↓ switch (device_type)
具体后端命名空间（infiniccl::cuda/ascend/kunlun/...）
    ↓ 类型转换 + 参数映射
厂商通信库（NCCL/HCCL/CNCL/BKCL/MCCL）
    ↓ 硬件抽象
网卡/PCIe/NVLink/HCCS 等物理互联
```

### 3.2 初始化流程对比

| 后端 | 设备设置要求 | 通信器初始化 API | 特殊参数 |
|------|-------------|-----------------|---------|
| cuda | 无需预设 | `ncclCommInitAll` | 标准模式 |
| ascend | **逆向遍历** `aclrtSetDevice` | `HcclCommInitAll` | device_ids 需转为 int32_t* |
| kunlun | 无需预设 | `bkcl_comm_init_all` | 直接使用 device_ids |
| cambricon | **正向** `cnrtSetDevice` | `cnclInitComms` | **额外需 rank_list 和 ndevice** |
| metax | 无需预设 | `hcclCommInitAll` | 标准模式 |
| moore | 无需预设 | `mcclCommInitAll` | 标准模式 |

**关键差异**：
- **Ascend** 要求在初始化前对所有设备调用 `aclrtSetDevice`，且代码中采用**逆向遍历**（`for (int i = ndevice - 1; i >= 0; i--)`），可能为了满足 HCCL 的特定设备绑定顺序要求
- **Cambricon** 的 `cnclInitComms` 接口最复杂，需显式传入 rank_id 数组和总设备数

### 3.3 AllReduce 执行流程

所有后端的 AllReduce 执行遵循统一模式：

1. **类型检查**：验证数据类型是否为 F32/F16/BF16（Moore 不支持 BF16）
2. **枚举映射**：将 Infini 的 `infiniDtype_t` 和 `infinicclReduceOp_t` 转换为厂商特定枚举
3. **Stream 转换**：将 `infinirtStream_t`（运行时抽象）转换为具体 stream 类型
4. **调用厂商 API**：执行实际的集合通信操作
5. **错误检查**：通过 `CHECK_*` 宏验证返回值

### 3.4 设备复用策略（共享后端）

| 共享后端 | 设备类型 | 原因 |
|---------|---------|------|
| cuda | NVIDIA, ILUVATAR, QY, HYGON | 四者均兼容 CUDA 生态和 NCCL 接口 |
| metax | 沐曦（双模式） | 可选择 HCCL 兼容模式或原生 MCCL |

### 3.5 编译时隔离机制

每个后端的头文件（如 `infiniccl_cuda.h`）都包含条件编译逻辑：

```cpp
#if defined(ENABLE_XXX_API) && defined(ENABLE_CCL)
    INFINICCL_DEVICE_API_IMPL(xxx)  // 实际实现
#else
    INFINICCL_DEVICE_API_NOOP(xxx)  // 空实现，返回 NOT_SUPPORTED
#endif
```

这种设计允许：
- 同一份源码树通过不同的编译选项组合，生成支持不同硬件集合的二进制
- 未启用某个后端时，相关代码不产生实际符号，避免链接错误
- 所有后端的函数签名在编译期保证一致

### 3.6 数据流向

```
训练框架构建 AllReduce 请求
    ↓
infinicclAllReduce(sendbuf, recvbuf, count, dtype, op, comm, stream)
    ↓
分发器根据 comm->device_type 路由
    ↓
后端 allReduce 实现：
    - getNcclDtype(dtype) / getAscendDtype(dtype) / ...
    - getNcclRedOp(op) / getHcclRedOp(op) / ...
    - getCudaStream(stream) / getAscendStream(stream) / ...
    - ncclAllReduce(..., ncclComm, cudaStream)
    ↓
厂商库在设备间执行数据交换
    ↓
结果写入 recvbuf，用户代码继续执行
```

---

## 4. 设计优势与局限性

### 4.1 优势
1. **零运行时开销**：通过命名空间和编译时分派，避免虚函数表查找
2. **类型安全**：编译期强制所有后端实现相同接口
3. **可扩展性**：添加新硬件后端仅需新增目录和实现三个函数
4. **条件编译友好**：支持灵活的编译配置，避免依赖冲突

### 4.2 局限性
1. **仅支持 AllReduce**：当前未暴露 Broadcast、ReduceScatter、AllGather 等其他集合通信原语
2. **编译时后端选择**：不支持运行时动态加载后端（如通过 dlopen）
3. **硬编码错误处理**：不支持的类型直接 `std::abort()`，缺少优雅降级
4. **同步/异步混淆**：API 传入 stream 但未明确说明是否支持异步执行

---

## 5. 依赖关系图

```
infiniccl (本目录)
    ├─→ infinirt (运行时层)
    │    └─→ infinirtStream_t, infiniDevice_t, infiniDtype_t
    ├─→ 厂商通信库（外部依赖）
    │    ├─→ NCCL (NVIDIA/天数/青云/海光)
    │    ├─→ HCCL (昇腾/沐曦兼容模式)
    │    ├─→ CNCL (寒武纪)
    │    ├─→ BKCL (昆仑)
    │    └─→ MCCL (摩尔线程/沐曦原生)
    └─→ utils.h (内部工具)
         └─→ CHECK_INTERNAL, CHECK_DTYPE 宏
```

---

## 6. 扩展指南

若要添加新的硬件后端（如 AMD ROCm 的 RCCL），需按以下步骤操作：

1. **创建目录**：`infiniccl/rocm/`
2. **实现头文件**：`infiniccl_rocm.h` 使用 `INFINICCL_DEVICE_API_IMPL(rocm)` 宏
3. **实现源文件**：`infiniccl_rocm.cc` 实现三个函数：
   - `commInitAll`：调用 `rcclCommInitAll`，封装为 `InfinicclComm`
   - `commDestroy`：调用 `rcclCommDestroy`
   - `allReduce`：实现类型映射，调用 `rcclAllReduce`
4. **注册分发器**：在 `infiniccl.cc` 中添加 `COMM_INIT_ALL(INFINI_DEVICE_ROCM, rocm)` 等 case
5. **添加编译选项**：在构建系统中定义 `ENABLE_ROCM_API` 宏

---

## 总结

infiniccl 是一个典型的**桥接模式（Bridge Pattern）** 实现，将硬件相关的通信库接口与上层的分布式训练逻辑解耦。通过严格的接口契约和编译时分发机制，它在保持高性能的同时，为 Infini 框架提供了广泛的硬件兼容性。这种设计值得作为其他跨平台子系统的参考模板。
