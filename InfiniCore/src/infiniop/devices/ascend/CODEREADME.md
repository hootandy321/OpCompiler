# Ascend Device Backend Core Implementation Documentation

华为昇腾 (Ascend NPU) 设备后端实现，为 Infini 框架提供完整的华为昇腾 AI 处理器支持。该模块实现了设备句柄管理、张量描述符转换以及 AscendC 自定义内核的通用基础设施。

## 1. Module Structure

- **`ascend_handle.h`**: 设备句柄类声明，继承自统一的 `InfiniopHandle` 基类
- **`ascend_handle.cc`**: 设备句柄实现，负责创建昇腾设备实例
- **`ascend_kernel_common.h`**: AscendC 内核开发的通用常量与工具函数模板
- **`common_ascend.h`**: 公共数据结构与函数声明，包括张量描述符包装类
- **`common_ascend.cc`**: 张量描述符实现与数据类型转换逻辑
- **`CMakeLists.txt`**: AscendC 内核编译配置，使用华为 CANN 工具链

## 2. Core Classes

### `device::ascend::Handle`
- **Location**: `ascend_handle.h` / `ascend_handle.cc`
- **Primary Function**: 昇腾设备句柄类，管理设备标识与设备类型注册，作为所有昇腾操作的入口点
- **Key Members**:
  - 继承自 `InfiniopHandle`，通过基类构造函数初始化设备类型为 `INFINI_DEVICE_ASCEND` 和设备 ID
- **Core Methods**:
  - `Handle(int device_id)`: 构造函数，将设备类型设为昇腾并记录设备 ID
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 静态工厂方法，分配并初始化设备句柄，返回 `INFINI_STATUS_SUCCESS`
- **Lifecycle**: 使用堆分配 (`new Handle`) 创建，由调用方负责生命周期管理

### `aclnnTensorDescriptor`
- **Location**: `common_ascend.h` / `common_ascend.cc`
- **Primary Function**: 华为 CANN ACL 神经网络张量描述符的 RAII 包装类，自动管理底层 `aclTensor` 的创建与销毁
- **Key Members**:
  - `ndim: uint64_t`: 张量维度数
  - `shape: std::vector<int64_t>`: 张量形状（各维度大小）
  - `strides: std::vector<int64_t>`: 步幅（各维度的字节/元素偏移量）
  - `offset: int64_t`: 起始偏移量（默认 0）
  - `dataType: aclDataType`: ACL 数据类型枚举（如 ACL_FLOAT, ACL_INT32 等）
  - `format: aclFormat`: 内存布局格式（当前硬编码为 ACL_FORMAT_ND，即 n 维通用格式）
  - `storageShape: std::vector<int64_t>`: 物理存储形状，由步幅推断得出的 1D 缓冲区大小
  - `storageNdim: int64_t`: 存储维度数（通常为 1）
  - `tensor: aclTensor*`: 底层 ACL 张量句柄，由 `aclCreateTensor` 创建
- **Core Methods**:
  - `aclnnTensorDescriptor(infiniopTensorDescriptor_t desc, void *data)`: 从 Infini 框架张量描述符构造，自动推断存储形状并创建 ACL 张量，时间复杂度 O(ndim)
  - `aclnnTensorDescriptor(aclDataType dtype, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides, void *data)`: 直接从 ACL 类型构造，支持标量（ndim=0）和张量
  - `~aclnnTensorDescriptor()`: 析构函数，调用 `aclDestroyTensor` 释放底层张量资源
  - `numel() const`: 返回张量元素总数（shape 各维度乘积），使用 `std::accumulate` 计算
  - `toString()`: 生成人类可读的调试信息字符串，包含 ndim、shape、strides、offset、dataType、format、storageShape
- **Lifecycle**: RAII 模式，构造时调用 `aclCreateTensor`，析构时自动调用 `aclDestroyTensor`，防止资源泄漏

### Helper Functions & Templates

#### `alignTileLen<T>(size_t tile_len, size_t byte_align)`
- **Location**: `ascend_kernel_common.h`
- **Function**: 将给定的元素长度按字节对齐要求向上取整，返回对齐后的元素数量
- **Algorithm**: 计算字节大小，如果未对齐则补齐到下一个对齐边界，再转换回元素数量
- **Complexity**: O(1)
- **Typical Usage**: 在 AscendC 内核中将数据块对齐到 32 字节边界（`BYTE_ALIGN = 32`）以优化内存访问

#### `inferStorageShape(std::vector<int64_t> shape, std::vector<int64_t> strides)`
- **Location**: `common_ascend.cc`
- **Function**: 根据逻辑形状和步幅推断物理存储形状（1D 缓冲区大小）
- **Algorithm**: 计算 `max_offset = sum((shape[i] - 1) * strides[i])`，返回 `{max_offset + 1}`，确保覆盖所有访问的元素
- **Complexity**: O(ndim)
- **Error Handling**: 如果 shape 和 strides 长度不匹配，抛出 `std::invalid_argument` 异常

#### `toAclDataType(infiniDtype_t dt)`
- **Location**: `common_ascend.cc`
- **Function**: 将 Infini 框架的数据类型枚举转换为华为 ACL 对应的数据类型枚举
- **Supported Types**: I8/I16/I32/I64, U8/U16/U32/U64, F16/BF16/F32/F64
- **Return**: 对于不支持的类型返回 `ACL_DT_UNDEFINED`

## 3. API Interface

### Device Handle API

```cpp
namespace device::ascend {

class Handle : public InfiniopHandle {
    // 构造函数（私有）
    Handle(int device_id);

public:
    // 创建昇腾设备句柄的静态工厂方法
    // 参数:
    //   handle_ptr: 输出参数，返回新创建的设备句柄指针
    //   device_id: 昇腾设备 ID（通常为 0）
    // 返回: INFINI_STATUS_SUCCESS 表示成功
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
};

}
```

### Tensor Descriptor Conversion API

```cpp
// ACL 张量描述符包装类
struct aclnnTensorDescriptor {
    // 从 Infini 张量描述符构造
    aclnnTensorDescriptor(infiniopTensorDescriptor_t desc, void *data = nullptr);

    // 从 ACL 原始类型构造
    aclnnTensorDescriptor(
        aclDataType dtype,
        const std::vector<int64_t> &shape,
        const std::vector<int64_t> &strides,
        void *data = nullptr
    );

    // 析构函数（自动释放 ACL 张量）
    ~aclnnTensorDescriptor();

    // 计算张量元素总数
    size_t numel() const;

    // 生成调试字符串
    std::string toString();
};

// 数据类型转换函数
aclDataType toAclDataType(infiniDtype_t dt);
```

### Kernel Launch API (Example)

```cpp
// SwiGLU 内核启动函数（示例，定义于 ops/swiglu/ascend/）
extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,           // 输出/输入内存地址
    infiniDtype_t dtype,                  // 数据类型（F16 或 F32）
    size_t batch, size_t seq, size_t hd,  // 张量形状
    ptrdiff_t stride_batch_c,             // 输出步幅
    ptrdiff_t stride_batch_a,             // 输入 A 步幅
    ptrdiff_t stride_batch_b,             // 输入 B 步幅
    ptrdiff_t stride_seq_c,
    ptrdiff_t stride_seq_a,
    ptrdiff_t stride_seq_b,
    void *stream                          // ACL 流
);
```

### Error Handling Macros

```cpp
// ACL 错误检查宏（定义于 common_ascend.h）
#define CHECK_ACL(API) \
    CHECK_INTERNAL(API, ACL_SUCCESS)

// 获取最近一次 ACL 错误消息的宏
#define GetRecentErrMsg() \
    { \
        auto tmp_err_msg = aclGetRecentErrMsg(); \
        if (tmp_err_msg != NULL) { \
            printf(" ERROR Message : %s \n ", tmp_err_msg); \
        } \
    }
```

## 4. Usage Example

```cpp
// 示例：使用昇腾后端执行 SwiGLU 算子

#include "devices/ascend/ascend_handle.h"
#include "devices/ascend/common_ascend.h"
#include "ops/swiglu/ascend/swiglu_ascend_kernel.h"

// 1. 创建昇腾设备句柄
InfiniopHandle *handle = nullptr;
infiniStatus_t status = device::ascend::Handle::create(&handle, 0);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 2. 准备输入输出张量描述符
infiniopTensorDescriptor_t a_desc = /* ... */;
infiniopTensorDescriptor_t b_desc = /* ... */;
infiniopTensorDescriptor_t c_desc = /* ... */;

// 3. 转换为 ACL 张量描述符（用于 ACL NN API 调用）
aclnnTensorDescriptor acl_a(a_desc, a_data_ptr);
aclnnTensorDescriptor acl_b(b_desc, b_data_ptr);
aclnnTensorDescriptor acl_c(c_desc, c_data_ptr);

// 4. 使用自定义 AscendC 内核（替代 ACL NN API）
void *stream = /* 获取 ACL 流 */;
status = swiglu_kernel_launch(
    c_data_ptr, a_data_ptr, b_data_ptr,
    INFINI_DTYPE_F16,              // 半精度浮点
    batch_size, seq_len, hidden_dim,
    stride_batch_c, stride_batch_a, stride_batch_b,
    stride_seq_c, stride_seq_a, stride_seq_b,
    stream
);

// 5. ACL 张量描述符在析构时自动释放
// ACL 流需要显式同步
aclrtSynchronizeStream(stream);

// 6. 清理设备句柄
delete handle;
```

## 5. Implementation Details

### Memory Management
- **RAII 模式**: `aclnnTensorDescriptor` 使用 RAII（Resource Acquisition Is Initialization）确保 ACL 张量资源在析构时自动释放
- **堆分配**: 设备句柄通过 `new` 分配，调用方负责 `delete`
- **存储形状推断**: 使用 `inferStorageShape` 函数根据步幅计算 1D 物理缓冲区大小，确保覆盖所有逻辑索引访问

### Concurrency
- **ACL 流支持**: 所有内核启动函数接受 `void *stream` 参数（`aclrtStream`），支持异步执行和多流并发
- **无锁设计**: 当前实现未显式使用锁，依赖 CANN 运行时的线程安全保障
- **多核并行**: AscendC 内核使用 `BLOCK_NUM`（8）个 AI Core 并行执行，通过 `GetBlockIdx()` 获取当前核心 ID

### Performance
- **块级并行**: 自定义内核将隐藏维度（hidden_size）划分为 8 个块（`BLOCK_NUM`），每个 AI Core 处理一个块
- **内存对齐**: 使用 `alignTileLen` 函数将数据块对齐到 32 字节边界（`BYTE_ALIGN = 32`），优化 DMA 传输效率
- **双缓冲**: 使用 `BUFFER_NUM = 2` 实现计算与数据传输流水线重叠，减少 AI Core 空闲时间
- **数据类型优化**: 支持 F16/BF16 半精度以减少内存带宽压力，F32 用于需要高精度的场景

### Error Handling
- **ACL 状态码**: 所有 ACL API 调用通过 `CHECK_ACL` 宏检查，失败时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE` 等错误码
- **异常安全**: `inferStorageShape` 在输入不匹配时抛出 `std::invalid_argument` 异常
- **错误消息**: 使用 `GetRecentErrMsg()` 宏打印 CANN 运行时提供的详细错误信息

### Dependencies
- **华为 CANN 工具链**: 依赖 `ASCEND_TOOLKIT_HOME` 环境变量指向 CANN 安装目录
- **ACL (Ascend Computing Language)**: 提供张量管理 (`acl/acl.h`, `acl/acl_rt.h`) 和神经网络原语 (`aclnn/acl_meta.h`)
- **AscendC**: 华为提供的类 CUDA 编程模型，用于开发自定义内核（`__aicore__`, `__global__`, `GM_ADDR` 等关键字）
- **Infini 框架基础**: 依赖 `InfiniopHandle` 基类（`../../handle.h`）和张量描述符（`../../tensor.h`）

### Design Patterns
- **工厂模式**: `Handle::create()` 静态方法封装对象创建逻辑
- **包装器模式 (Wrapper)**: `aclnnTensorDescriptor` 将 C 风格的 `aclTensor` 包装为 C++ RAII 对象
- **模板方法模式**: 自定义内核使用模板 `SwigluKernel<T>` 支持多种数据类型（`half`, `float`）
- **宏生成代码**: `DEFINE_SWIGLU_KERNEL` 和 `LAUNCH_SWIGLU_KERNEL` 宏减少重复代码，为不同数据类型生成特化版本
- **策略模式**: 通过 `toAclDataType` 函数实现框架数据类型到硬件类型的映射策略

### AscendC Kernel Architecture (Extended)

虽然本目录主要包含基础设施，但理解内核架构对于正确使用至关重要：

```cpp
// AscendC 内核的典型三阶段模式
template <typename T>
__aicore__ inline void process() {
    for (size_t i = 0; i < total_tiles; ++i) {
        copyIn(i);    // 从全局内存 (GM) 拷贝到本地内存 (Local Tensor)
        compute(i);   // 在矢量计算单元 (Vector Unit) 执行计算
        copyOut(i);   // 从本地内存拷贝回全局内存
    }
}

// 队列管理（双缓冲流水线）
TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue_a, _in_queue_b;  // 输入队列
TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue_c;             // 输出队列
TPipe _pipe;                                                     // 流水线管理器

// 内存分配
_pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T));
```

### Build System (CMake)

- **SOC Version**: 默认配置为 `Ascend910B3`，可通过 CMake 缓存变量修改
- **工具链检测**: 自动搜索 CANN 包中的 `ascendc_kernel_cmake` 目录，支持多种安装路径（`tools/tikcpp`, `compiler/tikcpp`, `ascendc_devkit`）
- **静态库**: 编译 `ascend_kernels` 静态库，包含 SwiGLU、RoPE、RandomSample 等自定义内核
- **包含路径**: 自动添加 `../../../../include/infiniop/` 以访问框架头文件

### Limitations & Future Work

- **格式限制**: 当前仅支持 `ACL_FORMAT_ND`（通用 n 维格式），不支持 NC1HWC0、FRACTAL_Z 等 NPU 优化格式
- **数据类型覆盖**: 缺少对 BOOL、COMPLEX64、INT4 等特殊类型的支持
- **零尺寸张量**: `inferStorageShape` 对标量（ndim=0）有特殊处理，但其他边界情况可能需要完善
- **异步流管理**: 当前示例未展示流池（Stream Pool）或事件（Event）同步，实际生产环境需要更完善的流管理策略
