# `Paged Caching (NVIDIA)` 分页缓存核心实现文档

本模块实现了 NVIDIA GPU 后端的分页缓存操作（Paged Caching），这是大模型推理中 KV Cache 管理的关键优化技术。该模块负责将输入的 Key-Value 对写入分页管理的缓存块中，支持 F16/BF16/F32 数据类型，并针对 NVIDIA GPU 架构进行了线程块级别的性能优化。

## 1. 模块结构

- **`paged_caching_nvidia.cuh`**: 头文件，定义 NVIDIA 后端描述符宏，包含基础接口声明
- **`paged_caching_nvidia.cu`**: 核心实现，包含描述符类、内核启动逻辑、设备能力适配

## 2. 核心类

### `Descriptor`
- **位置**: `paged_caching_nvidia.cu` (namespace `op::paged_caching::nvidia`)
- **主要功能**: 封装 NVIDIA GPU 后端的分页缓存操作符，提供设备特定的内核调度和执行接口
- **关键成员**:
  - `_opaque`: PIMPL 指针，指向 `Opaque` 结构体（持有 `device::nvidia::Handle::Internal` 共享指针，用于访问 GPU 设备信息和能力）
  - `_info`: `PagedCachingInfo` 实例，存储张量形状、步长、数据类型等元数据
- **核心方法**:
  - `create(handle, desc_ptr, ...)`: 静态工厂方法，创建描述符实例。首先调用 `PagedCachingInfo::create()` 验证输入张量描述符的一致性（返回 `Result<PagedCachingInfo, infiniStatus_t>`），然后构造 Descriptor 并初始化 Opaque 对象（从传入的 handle 提取内部设备句柄）
  - `calculate(workspace, k_cache, v_cache, k, v, slot_mapping, stream_)`: 执行分页缓存写入操作。根据 GPU 架构的 `maxThreadsPerBlock` 能力选择最优的内核配置（1024/512/4096 线程），然后调用 `launchKernel<>()` 模板函数发起 CUDA 内核调用
- **生命周期**:
  - 构造: 通过 `create()` 静态方法分配堆内存，初始化 Opaque 内部状态
  - 析构: 释放 `_opaque` 指针（使用 `delete`，`std::shared_ptr` 自动管理设备句柄生命周期）

### `Opaque` (PIMPL 结构体)
- **位置**: `paged_caching_nvidia.cu` (Descriptor 内嵌定义)
- **主要功能**: 隐藏 NVIDIA 设备相关的实现细节，避免头文件暴露 CUDA 类型
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>`，持有 NVIDIA 设备句柄的内部实现（提供 `maxThreadsPerBlock()` 等设备能力查询接口）

## 3. API 接口

```cpp
// 描述符创建接口（工厂方法）
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                    // 全局 InfiniOP 句柄
    Descriptor **desc_ptr,                      // 输出：创建的描述符指针
    infiniopTensorDescriptor_t k_cache_desc,    // K Cache 张量描述符（分页缓存）
    infiniopTensorDescriptor_t v_cache_desc,    // V Cache 张量描述符（分页缓存）
    infiniopTensorDescriptor_t k_desc,          // 输入 K 张量描述符
    infiniopTensorDescriptor_t v_desc,          // 输入 V 张量描述符
    infiniopTensorDescriptor_t slot_mapping_desc // Slot 映射张量描述符（int64_t，映射 token 到缓存块）
);
// 返回值：成功返回 INFINI_STATUS_SUCCESS，失败返回错误码（张量不匹配、数据类型不支持等）

// 分页缓存执行接口
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,  // 工作空间缓冲区及大小（当前未使用）
    void *k_cache, void *v_cache,            // 输出：分页管理的 K/V Cache（设备内存）
    const void *k, const void *v,            // 输入：待写入的 K/V 张量（设备内存）
    const void *slot_mapping,                // 输入：token 到缓存块的映射（int64_t 数组）
    void *stream_                            // CUDA 流（cudaStream_t）
) const;
// 返回值：成功返回 INFINI_STATUS_SUCCESS，失败返回 INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED（GPU 不支持）或 INFINI_STATUS_BAD_TENSOR_DTYPE（数据类型不支持）
```

## 4. 使用示例

```cpp
// 示例：使用 NVIDIA 分页缓存操作符写入 KV Cache
// 假设已有：
//   - infiniopHandle_t handle（已初始化的 InfiniOP 句柄）
//   - k_cache_desc, v_cache_desc（描述分页缓存的形状和布局）
//   - k_desc, v_desc（描述输入 K/V 的形状和布局）
//   - slot_mapping_desc（描述 slot_mapping 的形状）
//   - 设备内存指针：d_k_cache, d_v_cache, d_k, d_v, d_slot_mapping
//   - CUDA 流：stream

// 1. 创建描述符
op::paged_caching::nvidia::Descriptor* paged_cache_desc;
infiniStatus_t status = op::paged_caching::nvidia::Descriptor::create(
    handle,
    &paged_cache_desc,
    k_cache_desc,
    v_cache_desc,
    k_desc,
    v_desc,
    slot_mapping_desc
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（张量描述符不匹配或数据类型不支持）
}

// 2. 执行分页缓存写入（将 K/V 写入分页管理的缓存块）
status = paged_cache_desc->calculate(
    nullptr, 0,     // 不需要额外工作空间
    d_k_cache, d_v_cache,  // 输出：分页缓存（设备内存）
    d_k, d_v,               // 输入：当前步骤的 K/V（设备内存）
    d_slot_mapping,         // 输入：token 到缓存块的映射（设备内存）
    (void*)stream           // CUDA 流
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（GPU 架构不支持或内核启动失败）
}

// 3. 清理资源（析构会自动释放 Opaque 内部状态）
delete paged_cache_desc;
```

## 5. 实现细节

- **内存管理**:
  - 使用 PIMPL 模式隔离 CUDA 依赖（`Opaque` 结构体持有 `std::shared_ptr<device::nvidia::Handle::Internal>`）
  - 描述符通过 `new` 分配堆内存，调用方需手动 `delete` 释放
  - 设备内存（k_cache, v_cache, k, v, slot_mapping）由调用方管理，本模块不负责分配或释放

- **并发性**:
  - 内核执行通过 CUDA 流（`cudaStream_t`）控制，支持多个流并发执行
  - 网格维度配置为 `(num_kv_heads, num_tokens, 1)`，每个 token-head 对由一个线程块处理
  - 块维度根据 GPU 能力选择（1024/512/4096 线程），所有线程协同处理单个 head 的所有维度（head_size * block_size）

- **性能**:
  - **内核调度策略**: 基于设备能力的运行时调度，通过 `maxThreadsPerBlock()` 查询选择最优配置（优先 1024 线程，次选 512/4096），确保在不同架构（A100/H100/V100/RTX 4090）上都能获得最优性能
  - **算法复杂度**: O(num_tokens * num_kv_heads * head_size * block_size)，每个线程块处理一个 token 的一个 head，线程数等于 head_size * block_size（理想情况）
  - **模板特化**: `launchKernel<NUM_THREADS>()` 为不同数据类型（F16/BF16/F32）和不同线程数生成特化内核，避免运行时分支开销
  - **无共享内存**: 内核不使用动态共享内存（`shared_mem_size = 0`），减少资源占用和启动开销

- **错误处理**:
  - **张量验证**: `PagedCachingInfo::create()` 返回 `Result<T, infiniStatus_t>`，检查输入张量维度、数据类型、步长的一致性
  - **设备兼容性**: `calculate()` 方法查询 `maxThreadsPerBlock()`，如果 GPU 不支持最小线程要求（512），返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
  - **数据类型支持**: 仅支持 `INFINI_DTYPE_F16`、`INFINI_DTYPE_BF16`、`INFINI_DTYPE_F32`，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

- **依赖**:
  - **外部模块**: `../paged_caching.h`（基础分页缓存接口定义）、`../../../devices/nvidia/nvidia_common.cuh`（NVIDIA 设备通用定义）、`../../../devices/nvidia/nvidia_kernel_common.cuh`（NVIDIA 内核通用工具）、`../cuda/kernel.cuh`（CUDA 后端内核实现）
  - **CUDA 工具链**: 依赖 CUDA Runtime API（`<<<>>>` 内核启动语法、`dim3`、`cudaStream_t`）
  - **硬件要求**: NVIDIA GPU，计算能力需支持至少 512 线程/块（大多数现代 GPU 满足）

- **设计模式**:
  - **PIMPL (Pointer to Implementation)**: 通过 `Opaque` 结构体隐藏 CUDA 相关实现细节，避免头文件暴露 GPU 类型
  - **工厂方法**: `create()` 静态方法封装对象创建和验证逻辑
  - **策略模式**: 运行时根据设备能力选择不同的内核配置（NUM_THREADS 参数）
  - **模板元编程**: `launchKernel<>()` 和 `pagedCaching<>()` 使用模板实现编译期类型特化和性能优化
