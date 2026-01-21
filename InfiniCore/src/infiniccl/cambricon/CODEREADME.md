# Cambricon CNCL 集合通信后端实现文档

本模块实现了 InfiniCCL 框架在寒武纪(Cambricon)硬件平台上的集合通信后端，基于寒武纪通信库(CNCL, Cambricon Collective Communication Library)提供多设备间的 AllReduce 等集合通信操作。

## 1. 模块结构

- **`infiniccl_cambricon.h`**: 头文件，根据编译时宏定义选择实现或空操作(noop)版本
- **`infiniccl_cambricon.cc`**: 核心实现文件，封装 CNCL API 提供通信器初始化、销毁和 AllReduce 操作

## 2. 核心数据结构与类型映射

### 2.1 InfinicclComm 结构
- **位置**: `infiniccl_impl.h` (父目录定义)
- **定义**:
```cpp
struct InfinicclComm {
    infiniDevice_t device_type;  // 设备类型(此处为 INFINI_DEVICE_CAMBRICON)
    int device_id;                // 实际的设备ID(非rank编号)
    void *comm;                   // CNCL通信器的原始指针(cnclComm_t)
};
```
- **作用**: 封装底层 CNCL 通信器，提供设备类型标识和设备ID映射

### 2.2 类型转换函数

#### `getCambriconStream(infinirtStream_t stream)`
- **功能**: 将抽象的 InfiniRT 流指针转换为 Cambricon CNRT 队列类型
- **实现**:
```cpp
inline cnrtQueue_t getCambriconStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return (cnrtQueue_t)(0);  // 空指针转零值队列
    }
    return static_cast<cnrtQueue_t>(stream);  // 直接类型转换
}
```
- **特殊处理**: 空流指针返回 0 而非 nullptr，符合 CNCL 规范

#### `getCnclComm(infinicclComm_t comm)`
- **功能**: 从通用通信器句柄提取底层 CNCL 通信器
- **实现**: `return static_cast<cnclComm_t>(comm->comm);`
- **类型安全**: 通过 void* 指针存储，使用时强制转换回 cnclComm_t

#### `getCnclDtype(infiniDtype_t datatype)`
- **功能**: 将 Infini 框架的通用数据类型枚举映射到 CNCL 数据类型
- **支持类型映射**:
  - `INFINI_DTYPE_F32` → `cnclFloat32`
  - `INFINI_DTYPE_F16` → `cnclFloat16`
  - `INFINI_DTYPE_BF16` → `cnclBfloat16`
- **错误处理**: 不支持的数据类型会输出错误信息并调用 `std::abort()` 终止程序
- **复杂度**: O(1) 常量时间查找表

#### `getCnclRedOp(infinicclReduceOp_t op)`
- **功能**: 将抽象的归约操作枚举映射到 CNCL 归约操作
- **操作映射**:
  - `INFINICCL_SUM` → `cnclSum`
  - `INFINICCL_PROD` → `cnclProd`
  - `INFINICCL_MAX` → `cnclMax`
  - `INFINICCL_MIN` → `cnclMin`
- **错误处理**: 不支持的操作类型会直接 `abort()`，未实现 `INFINICCL_AVG`
- **复杂度**: O(1) 常量时间查找表

## 3. API 接口实现

### 3.1 `commInitAll` - 初始化通信器组

**函数签名**:
```cpp
infiniStatus_t infiniccl::cambricon::commInitAll(
    infinicclComm_t *comms,      // [输出] 通信器数组指针
    int ndevice,                  // [输入] 设备数量
    const int *device_ids)        // [输入] 设备ID数组
```

**功能**: 初始化一组 Cambricon 设备间的 CNCL 通信器

**算法流程**:
1. **内存分配**: 创建 `std::vector<cnclComm_t>` 和 `std::vector<int>` 存储底层通信器和 rank 列表
2. **设备设置**: 遍历所有设备，对每个设备调用 `cnrtSetDevice(device_ids[i])` 设置当前设备上下文
3. **通信器初始化**: 调用 `cnclInitComms()` 批量初始化通信器
   - 参数: 通信器数组、设备数量、设备ID列表、rank列表、rank数量、根指针(nullptr)
   - CNCL 会自动建立设备间的通信拓扑(如 PCIe、NVLink 等高速互联)
4. **句柄封装**: 为每个设备创建 `InfinicclComm` 结构，封装:
   - 设备类型: `INFINI_DEVICE_CAMBRICON`
   - 设备ID: 实际物理设备号
   - 通信器指针: CNCL 返回的原始句柄
5. **返回状态**: 成功返回 `INFINI_STATUS_SUCCESS`

**错误处理**:
- `CHECK_INTERNAL(cnrtSetDevice(...), CNRT_RET_SUCCESS)`: 设备设置失败时返回 `INFINI_STATUS_INTERNAL_ERROR`
- `CHECK_CNCL(cnclInitComms(...))`: CNCL 初始化失败时返回 `INFINI_STATUS_INTERNAL_ERROR`

**复杂度**: O(n) n 为设备数量

**线程安全性**: 依赖 CNCL 库的实现，通常要求在调用前设置正确的设备上下文

**内存管理**: 使用 `new` 分配 `InfinicclComm`，需在销毁时手动释放

### 3.2 `commDestroy` - 销毁通信器

**函数签名**:
```cpp
infiniStatus_t infiniccl::cambricon::commDestroy(infinicclComm_t comm)
```

**功能**: 释放单个 CNCL 通信器及其资源

**算法流程**:
1. **释放 CNCL 通信器**: 调用 `cnclFreeComm(getCnclComm(comm))` 释放底层通信资源
2. **释放句柄**: 调用 `delete comm` 释放 `InfinicclComm` 结构体内存
3. **返回状态**: 成功返回 `INFINI_STATUS_SUCCESS`

**错误处理**:
- `CHECK_CNCL(cnclFreeComm(...))`: CNCL 释放失败时返回 `INFINI_STATUS_INTERNAL_ERROR`

**资源清理**: 确保 CNCL 内部通信队列、锁等资源正确释放，避免内存泄漏

### 3.3 `allReduce` - 全局归约并广播

**函数签名**:
```cpp
infiniStatus_t infiniccl::cambricon::allReduce(
    void *sendbuf,                // [输入] 发送缓冲区指针
    void *recvbuf,                // [输出] 接收缓冲区指针
    size_t count,                 // [输入] 元素数量
    infiniDtype_t datatype,       // [输入] 数据类型
    infinicclReduceOp_t op,       // [输入] 归约操作
    infinicclComm_t comm,         // [输入] 通信器句柄
    infinirtStream_t stream)      // [输入] 执行流(可为 nullptr)
```

**功能**: 在所有参与设备间执行归约操作，并将结果广播到所有设备

**算法流程**:
1. **数据类型验证**: 使用 `CHECK_DTYPE` 宏确保数据类型为 F32/F16/BF16 之一
   - 不支持类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
2. **类型转换**:
   - `getCnclDtype(datatype)`: 映射到 CNCL 数据类型
   - `getCnclRedOp(op)`: 映射到 CNCL 归约操作
   - `getCnclComm(comm)`: 提取底层通信器
   - `getCambriconStream(stream)`: 转换流指针
3. **调用 CNCL**: 执行 `cnclAllReduce()` 触发异步集合通信
4. **返回状态**: 成功返回 `INFINI_STATUS_SUCCESS`

**错误处理**:
- `CHECK_DTYPE()`: 数据类型不支持时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- `CHECK_CNCL()`: CNCL 调用失败时返回 `INFINI_STATUS_INTERNAL_ERROR`

**执行模式**:
- **同步 vs 异步**: CNCL 的 AllReduce 操作通常在指定的 CNRT 队列上异步执行
- **空流处理**: 当 `stream == nullptr` 时，传递 0 值队列，CNCL 使用默认流

**性能特性**:
- **带宽限制**: 受限于 Cambricon 设备间互联带宽(如 MLU 系列的高速互联)
- **延迟模型**: 小消息延迟主导，大消息带宽主导
- **拓扑感知**: CNCL 自动优化通信路径(优先使用高速互联如 PCIe Gen4/5 或专用互联)

## 4. 使用示例

```cpp
#include "infiniccl.h"
#include <vector>

int main() {
    // 1. 初始化两个 Cambricon 设备
    const int num_devices = 2;
    const int device_ids[] = {0, 1};
    std::vector<infinicclComm_t> comms(num_devices);

    // 初始化 CNCL 通信器
    infinicclCommInitAll(
        INFINI_DEVICE_CAMBRICON,
        comms.data(),
        num_devices,
        device_ids
    );

    // 2. 准备数据(假设已在 MLU 内存中分配)
    size_t count = 1024;
    float *send_buf = nullptr;  // MLU 设备内存指针
    float *recv_buf = nullptr;  // MLU 设备内存指针
    // ... (省略内存分配和初始化)

    // 3. 执行 AllReduce 求和
    infinicclAllReduce(
        send_buf,              // 发送缓冲区
        recv_buf,              // 接收缓冲区
        count,                 // 元素数量
        INFINI_DTYPE_F32,      // 数据类型(32位浮点)
        INFINICCL_SUM,         // 归约操作(求和)
        comms[0],              // 通信器(设备0)
        nullptr                // 使用默认流
    );

    // 4. 同步等待(如果使用流)
    // cnrtQueueSync(...);

    // 5. 清理资源
    for (int i = 0; i < num_devices; i++) {
        infinicclCommDestroy(comms[i]);
    }

    return 0;
}
```

## 5. 实现细节

### 5.1 编译时条件编译

**宏控制逻辑**:
```cpp
#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_CCL)
    INFINICCL_DEVICE_API_IMPL(cambricon)  // 启用实现
#else
    INFINICCL_DEVICE_API_NOOP(cambricon)  // 空操作版本
#endif
```

**效果**:
- 未定义 `ENABLE_CAMBRICON_API` 或 `ENABLE_CCL` 时，所有 API 返回 `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`
- 允许在同一二进制中支持多种设备后端(CUDA、CPU、Cambricon 等)

### 5.2 错误处理宏

**CHECK_CNCL 宏定义**:
```cpp
#define CHECK_CNCL(API__) CHECK_INTERNAL(API__, CNCL_RET_SUCCESS)
```

**展开后行为**:
```cpp
do {
    auto api_result_ = (API__);
    if (api_result_ != CNCL_RET_SUCCESS) {
        std::cerr << "Error Code " << api_result_ << " in `" << #API__ << "`"
                  << " from " << __func__
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }
} while (0)
```

**优点**:
- 自动记录错误发生位置(文件名、行号、函数名)
- 统一的错误处理逻辑
- 避免嵌套 if-else，提高代码可读性

**CHECK_DTYPE 宏**:
- 验证数据类型在支持列表中
- 失败时打印类型名称并返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- 使用变参宏实现多类型检查

### 5.3 内存管理策略

**InfinicclComm 生命周期**:
1. **分配**: 在 `commInitAll` 中使用 `new InfinicclComm{}` 分配
2. **所有权**: 调用者拥有通信器句柄，负责调用 `commDestroy` 释放
3. **释放**: 在 `commDestroy` 中先释放 CNCL 资源，再 `delete` 结构体

**内存泄漏风险**:
- 如果 `commDestroy` 未被调用，`InfinicclComm` 和底层 `cnclComm_t` 会泄漏
- CNCL 通信器可能持有设备队列、锁等资源，泄漏会导致设备资源耗尽

**最佳实践**:
- 使用 RAII 包装器(未在此实现中提供)
- 或确保在异常发生时仍调用 `commDestroy`

### 5.4 并发与线程安全

**设备上下文设置**:
- `commInitAll` 中对每个设备调用 `cnrtSetDevice(device_ids[i])`
- CNCL 要求在操作前设置正确的设备上下文
- 多线程环境下需确保设备上下文不会被其他线程篡改

**通信器并发使用**:
- 同一通信器可在多个流上并发调用 AllReduce
- CNCL 内部使用锁保护通信器资源
- 不同流上的操作可能并行执行(取决于 CNCL 实现)

**线程安全保证**:
- 依赖 CNCL 库的线程安全性
- 通常要求每个线程使用不同的设备或通信器
- 共享通信器时需外部同步

### 5.5 性能优化考虑

**批量初始化**:
- `cnclInitComms` 一次性初始化所有设备通信器
- 相比逐个初始化减少开销，并允许 CNCL 优化全局拓扑

**流执行**:
- AllReduce 在指定流上异步执行，可与计算操作重叠
- 空流指针使用默认流，可能与显式管理的流产生同步问题

**数据类型支持**:
- 仅支持 F32/F16/BF16 三种浮点类型
- 不支持整数类型，可能限制某些应用场景(如梯度累积需要 int32)

### 5.6 依赖关系

**外部依赖**:
- **CNCL (Cambricon Collective Communication Library)**: 提供底层集合通信原语
- **CNRT (Cambricon Runtime)**: 提供设备管理和队列接口
- **InfiniRT**: 抽象的运行时接口，提供设备和流类型定义

**内部依赖**:
- `infiniccl_impl.h`: 定义 `InfinicclComm` 结构和 API 宏
- `utils.h`: 提供 `CHECK_INTERNAL` 和 `CHECK_DTYPE` 宏
- `infinicore.h`: 提供 `infiniDtype_t` 等基础类型定义

### 5.7 限制与未实现功能

**缺失功能**:
1. **INFINICCL_AVG 操作**: `getCnclRedOp` 中未实现平均操作
2. **整数类型支持**: `getCnclDtype` 仅支持浮点类型
3. **其他集合操作**: 未实现 Broadcast、Reduce、Scatter、Gather、AllGather 等
4. **通信组管理**: 不支持子通信组或笛卡尔拓扑

**错误处理限制**:
- 数据类型不支持时直接 `abort()`，而非优雅降级
- 缺少详细错误信息传播机制

## 6. 设计模式

**适配器模式 (Adapter Pattern)**:
- 将 CNCL 的 C 接口适配到 InfiniCCL 的抽象接口
- 类型转换函数充当适配器，统一不同后端的 API 差异

**工厂模式 (Factory Pattern)**:
- `commInitAll` 充当工厂方法，批量创建通信器对象
- 封装复杂的初始化逻辑，简化用户调用

**策略模式 (Strategy Pattern)**:
- 通过编译时宏选择实现策略(实现 vs noop)
- 允许在不修改代码的情况下切换后端支持

**RAII (资源获取即初始化)**:
- 虽未完全实现，但 `commInitAll`/`commDestroy` 配对遵循 RAII 思想
- 建议上层使用 C++ 智能指针包装 `InfinicclComm`
