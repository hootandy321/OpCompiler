# InfiniCCL Kunlun Backend Implementation Documentation

Kunlun (昆仑) 处理器后端实现,基于 BKCL (Baidu Kunlun Communication Library) 提供跨设备集合通信功能,支持 AllReduce 等集合通信原语。

## 1. Module Structure

- **`infiniccl_kunlun.h`**: 头文件,根据编译时宏条件定义 Kunlun API 接口或空操作实现
- **`infiniccl_kunlun.cc`**: 核心实现文件,提供 BKCL 通信接口的封装与类型转换逻辑

## 2. Core Components

### `InfinicclComm` 结构体
- **Location**: `infiniccl_impl.h` (父目录,被本模块引用)
- **Primary Function**: 通用通信器容器,用于抽象不同硬件后端的通信上下文
- **Key Members**:
  - `device_type: infiniDevice_t`: 设备类型标识符 (本模块中为 `INFINI_DEVICE_KUNLUN`)
  - `device_id: int`: 物理设备 ID (非 rank 编号)
  - `comm: void*`: 原始通信器指针 (本模块中存储 `BKCLContext_t`)
- **Lifecycle**: 通过 `commInitAll` 批量创建,通过 `commDestroy` 单独销毁

## 3. API Interface

### 通信器初始化与销毁

```cpp
namespace infiniccl::kunlun {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,    // [输出] 通信器数组指针
    int ndevice,               // [输入] 设备数量
    const int *device_ids);    // [输入] 设备 ID 数组
// 功能: 初始化指定 Kunlun 设备的 BKCL 通信上下文
// 返回: INFINI_STATUS_SUCCESS 成功,否则返回错误码

infiniStatus_t commDestroy(infinicclComm_t comm);
// 功能: 销毁单个通信器并释放 BKCL 上下文
// 返回: INFINI_STATUS_SUCCESS 成功
}
```

### 集合通信操作

```cpp
infiniStatus_t allReduce(
    void *sendbuf,                // [输入] 发送缓冲区指针 (设备内存)
    void *recvbuf,                // [输出] 接收缓冲区指针 (设备内存)
    size_t count,                 // [输入] 元素数量
    infiniDtype_t datatype,       // [输入] 数据类型 (支持 F32/F16/BF16)
    infinicclReduceOp_t op,       // [输入] 归约操作 (SUM/PROD/MAX/MIN)
    infinicclComm_t comm,         // [输入] 通信器
    infinirtStream_t stream);     // [输入] Kunlun 计算流 (可为 nullptr)
// 功能: 执行跨设备的 AllReduce 操作
// 返回: INFINI_STATUS_SUCCESS 成功
```

## 4. Usage Example

```cpp
#include "infiniccl/kunlun/infiniccl_kunlun.h"

// 初始化双卡通信
int device_ids[] = {0, 1};
infinicclComm_t comms[2];

// 创建 BKCL 通信上下文
infiniStatus_t status = infiniccl::kunlun::commInitAll(comms, 2, device_ids);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理初始化失败
}

// 准备数据 (假设已分配在 Kunlun 设备内存)
float *send_buf = allocate_device_buffer<float>(1024);
float *recv_buf = allocate_device_buffer<float>(1024);

// 执行 AllReduce (SUM 操作,在默认流上)
status = infiniccl::kunlun::allReduce(
    send_buf,
    recv_buf,
    1024,                           // 元素数量
    INFINI_DTYPE_F32,              // FP32 数据类型
    INFINICCL_SUM,                 // 求和归约
    comms[0],                      // 使用第 0 号通信器
    nullptr                        // 使用默认流
);

// 清理资源
infiniccl::kunlun::commDestroy(comms[0]);
infiniccl::kunlun::commDestroy(comms[1]);
```

## 5. Implementation Details

### 类型映射机制

**数据类型转换** (`getBkclDtype`):
```cpp
BKCLDataType getBkclDtype(infiniDtype_t datatype) {
    // INFINI_DTYPE_F32  → BKCL_FLOAT
    // INFINI_DTYPE_F16  → BKCL_FLOAT16
    // INFINI_DTYPE_BF16 → BKCL_BFLOAT16
    // 其他类型 → 调用 std::abort() 终止程序
}
```

**归约操作转换** (`getBkclRedOp`):
```cpp
BKCLOp getBkclRedOp(infinicclReduceOp_t op) {
    // INFINICCL_SUM  → BKCL_ADD
    // INFINICCL_PROD → BKCL_PRODUCT
    // INFINICCL_MAX  → BKCL_MAX
    // INFINICCL_MIN  → BKCL_MIN
    // 其他操作 → 调用 std::abort() 终止程序
}
```

### 流与通信器处理

**流类型转换** (`getKunlunStream`):
- 将 `infinirtStream_t` 转换为 Kunlun 原生 `XPUStream` 类型
- 空指针 (`nullptr`) 映射为 Kunlun 默认流 (值为 0)

**通信器提取** (`getBkclComm`):
- 从 `InfinicclComm` 包装器中提取原始 `BKCLContext_t` 指针
- 使用 `reinterpret_cast` 进行不安全但必要的类型转换

### 错误处理策略

**BKCL 错误检查**:
```cpp
#define CHECK_BKCL(API__) CHECK_INTERNAL(API__, BKCL_SUCCESS)
```
- 所有 BKCL API 调用都通过 `CHECK_BKCL` 宏包装
- 宏定义位于 `../../utils.h`,检查返回值是否等于 `BKCL_SUCCESS`
- 失败时触发错误处理流程 (可能抛出异常或返回错误码)

**数据类型验证**:
```cpp
CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
```
- 在 `allReduce` 入口处验证数据类型合法性
- 仅支持浮点类型 (FP32/FP16/BF16)

### 编译时条件化

头文件使用双重宏检查:
```cpp
#if defined(ENABLE_KUNLUN_API) && defined(ENABLE_CCL)
    INFINICCL_DEVICE_API_IMPL(kunlun)  // 提供真实实现
#else
    INFINICCL_DEVICE_API_NOOP(kunlun)  // 提供空操作 stub
#endif
```
- `ENABLE_KUNLUN_API`: 启用 Kunlun 设备支持
- `ENABLE_CCL`: 启用集合通信框架
- 两者均定义时才链接 BKCL 实现,否则所有函数返回 `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`

### 依赖项

**外部依赖**:
- `<bkcl.h>`: 百度昆仑通信库头文件,提供 BKCL 原生 API
- `../infiniccl_impl.h`: 通信器结构定义与接口宏
- `../../utils.h`: 工具宏 (`CHECK_INTERNAL`, `CHECK_DTYPE`)

**标准库**:
- `<iostream>`: 错误信息输出 (用于 `std::cerr`)
- `<vector>`: 动态数组 (用于 `commInitAll` 中的 `bkcl_comm_init_all`)
- `<cstdlib>`: 提供 `std::abort()` (隐式包含)

### 内存管理

- **通信器分配**: 使用 C++ `new` 分配 `InfinicclComm` 对象
- **批量初始化**: `commInitAll` 通过 `std::vector<bkclComm_t>` 临时存储 BKCL 上下文,然后包装为 `InfinicclComm` 数组
- **销毁顺序**: 先调用 `bkcl_destroy_context` 释放 BKCL 资源,再 `delete` 通信器对象

### 并发模型

- **流语义**: 支持异步执行,通过 `stream` 参数指定 Kunlun 计算流
- **线程安全性**: BKCL 库本身的线程安全属性取决于底层实现,本封装层不引入额外同步机制
- **默认流**: `stream == nullptr` 时使用 Kunlun 默认流 (值为 0),可能与显式创建的流同步执行

### 性能考量

- **零拷贝设计**: `sendbuf` 和 `recvbuf` 直接传递给 BKCL,避免中间缓冲
- **类型转换开销**: `getBkclDtype` 和 `getBkclRedOp` 使用简单的 `switch-case`,O(1) 复杂度
- **批量初始化**: `bkcl_comm_init_all` 一次性初始化所有设备通信器,减少环形网络建立开销
