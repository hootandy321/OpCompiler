# InfiniCCL CUDA Backend Core Implementation Documentation

InfiniCCL CUDA 后端实现是基于 NVIDIA NCCL (NVIDIA Collective Communications Library) 的封装层，为 Infini 框架提供跨 GPU 的集合通信原语，支持多设备张量归约操作。该模块作为 InfiniCCL 统一通信接口的 NVIDIA GPU 特定实现。

## 1. Module Structure

- **`infiniccl_cuda.h`**: 头文件，通过宏条件编译控制 CUDA 后端的启用，定义 `infiniccl::cuda` 命名空间下的 API 声明
- **`infiniccl_cuda.cu`**: CUDA C++ 实现文件，包含 NCCL 通信域管理、数据类型转换、归约操作映射以及 AllReduce 核心逻辑

## 2. Core Classes

### `InfinicclComm`
- **Location**: `../infiniccl_impl.h` (defined in parent directory)
- **Primary Function**: 抽象通信域句柄，封装设备类型和底层通信库的 communicator 对象
- **Key Members**:
  - `infiniDevice_t device_type`: 设备类型标识 (NVIDIA = 1)
  - `int device_id`: 物理设备 ID (非 rank 编号)
  - `void *comm`: 底层通信库 communicator 的不透明指针 (CUDA 下为 `ncclComm_t`)
- **Lifecycle**: 由 `commInitAll` 批量创建并分配，由 `commDestroy` 显式销毁并释放内存

### `infiniccl::cuda` Namespace Functions
- **Location**: `infiniccl_cuda.cu`
- **Primary Function**: 实现 NCCL 的设备特定封装，提供类型安全的转换层和错误处理
- **Core Functions**:
  - `commInitAll()`: 初始化多设备 NCCL 通信域
  - `commDestroy()`: 销毁 NCCL 通信域并释放资源
  - `allReduce()`: 执行跨设备 AllReduce 集合通信操作
- **Lifecycle**: 由上层 `infiniccl.cc` 根据 `device_type` 调度执行

## 3. API Interface

```cpp
namespace infiniccl::cuda {

// 初始化多设备 NCCL 通信域
// 参数:
//   comms: 输出参数，分配的 InfinicclComm 句柄数组
//   ndevice: 设备数量
//   device_ids: 设备 ID 数组 (物理 GPU 编号)
// 返回: INFINI_STATUS_SUCCESS 或错误码
infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);

// 销毁通信域
// 参数:
//   comm: 待销毁的通信域句柄
// 返回: INFINI_STATUS_SUCCESS 或错误码
infiniStatus_t commDestroy(infinicclComm_t comm);

// 执行 AllReduce 集合通信
// 参数:
//   sendbuf: 发送缓冲区指针 (GPU 内存)
//   recvbuf: 接收缓冲区指针 (GPU 内存，可与 sendbuf 相同实现 in-place)
//   count: 元素数量
//   datatype: 数据类型 (F32/F16/BF16)
//   op: 归约操作 (SUM/PROD/MAX/MIN/AVG)
//   comm: 通信域句柄
//   stream: CUDA 流 (可为 nullptr 使用默认流)
// 返回: INFINI_STATUS_SUCCESS 或错误码
infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

} // namespace infiniccl::cuda
```

## 4. Usage Example

```cpp
// 示例: 使用 InfiniCCL CUDA 后端执行多 GPU AllReduce
#include "infiniccl.h"

// 1. 初始化通信域 (假设有 4 个 GPU)
const int num_devices = 4;
int device_ids[] = {0, 1, 2, 3};

infinicclComm_t comms[num_devices];
infiniStatus_t status = infinicclCommInitAll(
    INFINI_DEVICE_NVIDIA,  // 设备类型
    comms,                 // 通信域句柄数组
    num_devices,           // 设备数量
    device_ids             // 设备 ID 数组
);

// 2. 分配 GPU 内存并初始化数据
size_t tensor_size = 1024 * 1024;  // 1M 元素
float *d_sendbuf, *d_recvbuf;

for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_sendbuf, tensor_size * sizeof(float));
    cudaMalloc(&d_recvbuf, tensor_size * sizeof(float));
    // ... 初始化 d_sendbuf 数据 ...
}

// 3. 创建 CUDA 流 (可选)
infinirtStream_t stream;
cudaStreamCreate((cudaStream_t*)&stream);

// 4. 执行 AllReduce 操作
status = infinicclAllReduce(
    d_sendbuf,              // 发送缓冲区
    d_recvbuf,              // 接收缓冲区
    tensor_size,            // 元素数量
    INFINI_DTYPE_F32,       // 数据类型: FP32
    INFINICCL_SUM,          // 归约操作: 求和
    comms[rank],            // 当前 rank 的通信域
    stream                  // CUDA 流
);

// 5. 同步并等待完成
cudaStreamSynchronize((cudaStream_t)stream);

// 6. 清理资源
for (int i = 0; i < num_devices; i++) {
    infinicclCommDestroy(comms[i]);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
}
```

## 5. Implementation Details

### Memory Management
- **通信域分配**: 使用 `new InfinicclComm{}` 在堆上分配通信域对象，生命周期由调用者管理
- **NCCL 资源**: NCCL communicator 由 `ncclCommInitAll` 批量创建，内部使用 NCCL 的内存池管理通信缓冲区
- **指针转换**: 使用 `static_cast` 和 `reinterpret_cast` 在类型安全前提下进行不透明指针转换

### Concurrency
- **CUDA 流支持**: 通过 `getCudaStream()` 转换层支持用户自定义 CUDA 流，`nullptr` 映射到 CUDA 默认流 (stream 0)
- **NCCL 线程模型**: NCCL 内部使用 CUDA 事件和流实现异步执行，确保与 CUDA kernels 的正确同步
- **无显式锁**: NCCL API 本身是线程安全的，本实现未引入额外锁机制

### Performance
- **批量初始化**: `ncclCommInitAll` 一次性初始化所有设备通信域，减少初始化开销
- **类型转换优化**: 使用 switch-case 实现 O(1) 复杂度的类型映射表
- **零拷贝**: AllReduce 支持 in-place 操作 (sendbuf == recvbuf)，NCCL 内部优化避免不必要的数据拷贝
- **支持的归约操作**: SUM, PROD, MAX, MIN, AVG (直接映射到 NCCL 原语，无额外开销)

### Error Handling
- **错误检查宏**: 使用 `CHECK_NCCL()` 宏封装 NCCL API 调用，失败时返回 `INFINI_STATUS_INTERNAL_ERROR`
- **数据类型验证**: 使用 `CHECK_DTYPE()` 宏在运行时验证支持的数据类型，不支持的返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **未处理路径**: 数据类型或归约操作的 default 分支调用 `std::abort()` 终止程序
- **空指针处理**: `getCudaStream()` 对 nullptr 返回 0 (CUDA 默认流)，`commDestroy()` 在 null comm 时返回成功

### Dependencies
- **外部依赖**:
  - `cuda_runtime.h`: CUDA 运行时 API
  - `nccl.h`: NVIDIA Collective Communications Library (版本 >= 2.0)
- **内部依赖**:
  - `../infiniccl_impl.h`: 定义 `InfinicclComm` 结构体和 `INFINICCL_DEVICE_API_IMPL` 宏
  - `../../utils.h`: 提供 `CHECK_NCCL`, `CHECK_DTYPE` 等错误检查宏
  - `include/infinirt.h`: 定义 `infinirtStream_t` 等运行时类型
- **编译条件**: 仅当定义 `ENABLE_NVIDIA_API`, `ENABLE_CCL` 且非 Windows 平台时编译实际实现，否则生成 noop 函数

### Design Patterns
- **Strategy Pattern**: 通过 `infiniccl::cuda` 命名空间封装 NCCL 特定策略，与其他后端 (kunlun/ascend/metax) 形成并行实现
- **Template Method Pattern**: 父级 `infiniccl.cc` 定义算法骨架，通过 switch-case 调用特定命名空间的实现
- **Adapter Pattern**: `getNcclDtype()`, `getNcclRedOp()`, `getCudaStream()` 等函数适配 Infini 类型到 NCCL 类型
- **RAII-lite**: `InfinicclComm` 使用裸指针手动管理，依赖调用者正确调用 `commDestroy` 释放资源
- **Compilation Guard**: 使用 `INFINICCL_DEVICE_API_IMPL/NOOP` 宏实现条件编译，支持无 NCCL 环境下的降级构建

### Hardware Backend Reuse
- **多 GPU 厂商支持**: 同一 CUDA 实现被复用于多个 NVIDIA 兼容硬件:
  - `INFINI_DEVICE_NVIDIA` (原厂 NVIDIA GPU)
  - `INFINI_DEVICE_ILUVATAR` (天数智芯 GPU)
  - `INFINI_DEVICE_QY` (QY GPU)
  - `INFINI_DEVICE_HYGON` (海光 GPU)
- **实现共享**: 这些设备类型在 `infiniccl.cc` 中均路由到 `infiniccl::cuda` 命名空间，避免代码重复
- **兼容性假设**: 假设这些厂商的驱动与 NCCL API 二进制兼容，实际行为依赖底层驱动实现
