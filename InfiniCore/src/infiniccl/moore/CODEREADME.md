# Moore CCL Backend Core Implementation Documentation

本模块实现了 InfiniCCL 通信库在摩尔线程（Moore Threads）GPU 硬件平台上的后端适配层，通过封装 MCCL（Moore Collective Communications Library）提供跨设备的集合通信原语，主要用于多 GPU 卡之间的 AllReduce 等集合通信操作。

## 1. Module Structure

- **`infiniccl_moore.h`**: 头文件，定义编译时条件编译逻辑，根据 `ENABLE_MOORE_API` 和 `ENABLE_CCL` 宏决定启用完整实现或空实现
- **`infiniccl_moore.cc`**: 核心实现文件，包含 MCCL API 的类型转换、通信器初始化/销毁、AllReduce 操作的完整封装

## 2. Core Classes

### `InfinicclComm`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniccl/infiniccl_impl.h`
- **Primary Function**: 通用通信器封装结构，存储设备类型、设备 ID 和底层通信库句柄的桥接结构
- **Key Members**:
  - `device_type`: `infiniDevice_t` 枚举，标识硬件设备类型（Moore 平台为 `INFINI_DEVICE_MOORE`）
  - `device_id`: `int` 类型，存储实际的物理设备 ID（非 rank 编号）
  - `comm`: `void*` 指针，指向底层 MCCL 通信器（`mcclComm_t`）的原始句柄
- **Lifecycle**: 由 `commInitAll` 批量构造并初始化，通过 `commDestroy` 显式销毁，采用堆分配（`new`/`delete`）

### `infiniccl::moore` Namespace Functions
- **Location**: `infiniccl_moore.cc`
- **Primary Function**: Moore 后端的实现命名空间，包含三个核心 API 函数，负责将 InfiniCCL 抽象接口映射到 MCCL 具体调用
- **Core Methods**:

  #### `commInitAll(infinicclComm_t *comms, int ndevice, const int *device_ids)`
  - **返回类型**: `infiniStatus_t`
  - **算法**: O(n) 批量初始化，先通过 `mcclCommInitAll` 一次性创建所有 MCCL 通信器，再遍历构造封装对象
  - **实现细节**:
    1. 创建 `std::vector<mcclComm_t>` 临时数组存储原生 MCCL 通信器
    2. 调用 `mcclCommInitAll(mccl_comms.data(), ndevice, device_ids)` 批量初始化
    3. 遍历 `[0, ndevice)` 为每个设备创建 `InfinicclComm` 封装对象，设置 `device_type = INFINI_DEVICE_MOORE`
    4. 错误处理通过 `CHECK_MCCL` 宏自动捕获 `mcclSuccess` 以外的错误码
  - **复杂度**: 时间 O(n)，空间 O(n)

  #### `commDestroy(infinicclComm_t comm)`
  - **返回类型**: `infiniStatus_t`
  - **实现细节**:
    1. 调用 `mcclCommDestroy` 销毁底层 MCCL 通信器
    2. 执行 `delete comm` 释放 `InfinicclComm` 封装对象内存
    3. 错误处理通过 `CHECK_MCCL` 宏确保 MCCL API 调用成功
  - **副作用**: 释放堆内存并销毁 MCCL 通信上下文

  #### `allReduce(void *sendbuf, void *recvbuf, size_t count, infiniDtype_t datatype, infinicclReduceOp_t op, infinicclComm_t comm, infinirtStream_t stream)`
  - **返回类型**: `infiniStatus_t`
  - **实现细节**:
    1. **类型校验**: 仅支持 `INFINI_DTYPE_F32` 和 `INFINI_DTYPE_F16`，其他类型返回 `INFINI_STATUS_BAD_PARAM`
    2. **类型转换**:
       - `infiniDtype_t` → `mcclDataType_t`（通过 `getMcclDtype` 函数）
       - `infinicclReduceOp_t` → `mcclRedOp_t`（通过 `getMcclRedOp` 函数）
       - `infinirtStream_t` → `musaStream_t`（通过 `getMusaStream` 函数，nullptr 映射为 0）
    3. **核心调用**: `mcclAllReduce(sendbuf, recvbuf, count, dtype, op, comm, stream)`
    4. 错误处理通过 `CHECK_MCCL` 宏
  - **内存语义**: 支持就地操作（sendbuf == recvbuf）和异地操作

### Type Conversion Functions
- **Location**: `infiniccl_moore.cc` (inline 函数)
- **Primary Function**: 在 Infini 抽象类型和 MCCL/MUSA 原生类型之间进行双向映射

  #### `getMusaStream(infinirtStream_t stream) → musaStream_t`
  - **映射规则**: `nullptr` → `0`（默认流），否则直接 `static_cast`

  #### `getMcclDtype(infiniDtype_t datatype) → mcclDataType_t`
  - **映射表**:
    - `INFINI_DTYPE_F32` → `mcclFloat`
    - `INFINI_DTYPE_F16` → `mcclHalf`
    - 其他类型 → `std::abort()` 终止程序

  #### `getMcclRedOp(infinicclReduceOp_t op) → mcclRedOp_t`
  - **映射表**:
    - `INFINICCL_SUM` → `mcclSum`
    - `INFINICCL_PROD` → `mcclProd`
    - `INFINICCL_MAX` → `mcclMax`
    - `INFINICCL_MIN` → `mcclMin`
    - `INFINICCL_AVG` → `mcclAvg`
    - 其他操作 → `std::abort()` 终止程序

  #### `getMcclComm(infinicclComm_t comm) → mcclComm_t`
  - **实现**: 从 `InfinicclComm::comm` 成员 `static_cast` 转换

## 3. API Interface

```cpp
// 通信器初始化 API（批量创建多个设备的通信器）
namespace infiniccl::moore {
infiniStatus_t commInitAll(
    infinicclComm_t *comms,      // [输出] 通信器指针数组，需预先分配 ndevice 个指针
    int ndevice,                 // [输入] 设备数量
    const int *device_ids);      // [输入] 设备 ID 数组，物理设备编号
// 返回: INFINI_STATUS_SUCCESS 成功
//      INFINI_STATUS_INTERNAL_ERROR MCCL 初始化失败

// 通信器销毁 API
infiniStatus_t commDestroy(
    infinicclComm_t comm);       // [输入] 要销毁的通信器
// 返回: INFINI_STATUS_SUCCESS 成功
//      INFINI_STATUS_INTERNAL_ERROR MCCL 销毁失败

// AllReduce 集合通信 API
infiniStatus_t allReduce(
    void *sendbuf,               // [输入] 发送缓冲区指针
    void *recvbuf,               // [输出] 接收缓冲区指针（可等于 sendbuf）
    size_t count,                // [输入] 元素数量（非字节数）
    infiniDtype_t datatype,      // [输入] 数据类型（仅支持 F32/F16）
    infinicclReduceOp_t op,      // [输入] 归约操作（SUM/PROD/MAX/MIN/AVG）
    infinicclComm_t comm,        // [输入] 通信器
    infinirtStream_t stream);    // [输入] MUSA 流（nullptr 表示默认流）
// 返回: INFINI_STATUS_SUCCESS 成功
//      INFINI_STATUS_BAD_PARAM 不支持的数据类型
//      INFINI_STATUS_INTERNAL_ERROR MCCL 执行失败
}
```

## 4. Usage Example

```cpp
#include "infiniccl/infiniccl.h"
#include "infinirt/infinirt.h"

// 初始化 4 张 Moore GPU 卡的通信环境
const int num_devices = 4;
int device_ids[4] = {0, 1, 2, 3};
infinicclComm_t comms[4] = {nullptr};

// 批量初始化通信器
auto status = infiniccl::moore::commInitAll(comms, num_devices, device_ids);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理初始化失败
}

// 在每张卡上准备数据（假设每张卡有 1024 个 float 元素）
size_t count = 1024;
float *send_buffer, *recv_buffer;
infinirtMalloc(reinterpret_cast<void **>(&send_buffer), count * sizeof(float));
infinirtMalloc(reinterpret_cast<void **>(&recv_buffer), count * sizeof(float));
// ... 填充 send_buffer 数据 ...

// 执行 AllReduce 操作（求和）
infinirtStream_t stream = nullptr; // 使用默认流
status = infiniccl::moore::allReduce(
    send_buffer,
    recv_buffer,
    count,
    INFINI_DTYPE_F32,
    INFINICCL_SUM,
    comms[0],  // 当前 rank 的通信器
    stream
);

// 同步并清理资源
infinirtStreamSynchronize(stream);
infinirtFree(send_buffer);
infinirtFree(recv_buffer);

// 销毁所有通信器
for (int i = 0; i < num_devices; ++i) {
    infiniccl::moore::commDestroy(comms[i]);
}
```

## 5. Implementation Details

### Memory Management
- **通信器对象分配**: 使用 C++ `new` 在堆上分配 `InfinicclComm` 对象，由用户负责调用 `commDestroy` 释放
- **MCCL 通信器生命周期**: 由 `mcclCommInitAll` 批量创建，通过 `mcclCommDestroy` 显式销毁
- **缓冲区管理**: 用户负责 `sendbuf`/`recvbuf` 的分配与释放（通过 `infinirtMalloc`/`infinirtFree`）

### Concurrency
- **流式执行**: 支持 MUSA 流（`musaStream_t`），允许与其他内核操作并发执行
- **线程安全**: 依赖 MCCL 库的线程安全保证，AllReduce 操作可在不同流中并发执行
- **通信器隔离**: 每个 `InfinicclComm` 对应独立的 MCCL 通信域，支持多进程/多线程独立使用

### Performance
- **批量初始化**: `commInitAll` 使用 `mcclCommInitAll` 一次性初始化所有设备，减少初始化开销
- **零拷贝语义**: AllReduce 支持 `sendbuf == recvbuf` 的就地操作，减少内存拷贝
- **类型限制**: 仅支持 `F16`/`F32` 两种精度，确保硬件加速路径

### Error Handling
- **错误检测宏**: `CHECK_MCCL(API__)` 展开为 `CHECK_INTERNAL(API__, mcclSuccess)`，自动检测 MCCL API 返回码
- **错误传播**: MCCL 错误转换为 `INFINI_STATUS_INTERNAL_ERROR`，类型错误返回 `INFINI_STATUS_BAD_PARAM`
- **终止策略**: 未知类型/操作触发 `std::abort()` 立即终止程序（快速失败原则）

### Dependencies
- **外部依赖**:
  - `<mccl.h>`: Moore Threads 集合通信库（MCCL）
  - `<musa_runtime.h>`: MUSA 运行时 API（提供 `musaStream_t` 类型）
- **内部依赖**:
  - `../infiniccl_impl.h`: 提供通信器封装结构和宏定义
  - `../../utils.h`: 提供 `CHECK_INTERNAL` 错误处理宏

### Design Patterns
- **Bridge Pattern**: `InfinicclComm` 作为桥接器，解耦抽象接口与 MCCL 具体实现
- **Strategy Pattern**: 编译时通过 `INFINICCL_DEVICE_API_IMPL`/`INFINICCL_DEVICE_API_NOOP` 宏选择完整实现或空实现
- **Template Method Pattern**: `infiniccl_impl.h` 中使用宏生成统一接口，各后端填充实现细节
- **RAII Lite**: 通信器采用显式创建/销毁（非 RAII），符合 C 风格 API 惯例

### Conditional Compilation
```cpp
#if defined(ENABLE_MOORE_API) && defined(ENABLE_CCL)
    INFINICCL_DEVICE_API_IMPL(moore)  // 完整实现
#else
    INFINICCL_DEVICE_API_NOOP(moore)  // 空实现（返回 NOT_SUPPORTED）
#endif
```
- **启用条件**: 必须同时定义 `ENABLE_MOORE_API` 和 `ENABLE_CCL` 宏
- **降级策略**: 未启用时所有函数返回 `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`

### Type Safety
- **严格类型检查**: `allReduce` 在运行时验证数据类型，拒绝非 F16/F32 的请求
- **类型转换安全**: 所有类型转换使用 `static_cast`，无 C 风格强制转换
- **枚举完整性**: `getMcclDtype`/`getMcclRedOp` 覆盖所有有效枚举值，未知值触发 `abort`

### Stream Handling
- **默认流支持**: `stream == nullptr` 映射到 MUSA 默认流（句柄 0）
- **显式流传递**: 支持 `infinirtStream_t` 到 `musaStream_t` 的透明转换
- **异步执行**: AllReduce 在指定流上异步执行，需显式同步以完成操作
