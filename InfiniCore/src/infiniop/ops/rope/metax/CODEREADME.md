# RoPE (Rotary Position Embedding) MetaX Backend Implementation Documentation

该模块实现了 Infini 框架中 RoPE (Rotary Position Embedding) 操作的 MetaX 硬件后端支持。RoPE 是 Transformer 模型中的关键位置编码技术,通过旋转角度编码位置信息到注意力机制中。该实现为 Moore Threads 的 MetaX GPU (如沐曦 MTG 系列) 提供了高性能的 CUDA 兼容计算内核。

## 1. Module Structure

- **`rope_metax.h`**: MetaX RoPE 操作符的公共头文件接口,定义了 DESCRIPTOR 宏以生成派生自 `InfiniopDescriptor` 的 `Descriptor` 类
- **`rope_metax.maca`**: 核心实现文件,包含 RoPE 操作符的 MetaX 设备内核封装、描述符创建、计算调度和类型分发逻辑

## 2. Core Classes

### `op::rope::metax::Descriptor`
- **Location**: `rope_metax.maca` (通过 `rope_metax.h` 中的 `DESCRIPTOR(metax)` 宏生成)
- **Primary Function**: 管理 MetaX 设备上 RoPE 操作的计算资源、验证输入张量形状、调度 GPU 内核执行。该类封装了 RoPE 计算的所有必要元数据和不透明设备句柄。
- **Key Members**:
  - `Opaque *_opaque`: 不透明指针,持有 MetaX 设备特定的内部状态(设备句柄、内核参数等)
  - `RoPEInfo _info`: RoPE 操作的静态元数据,包括张量形状、步长、数据类型、位置编码类型等
  - `size_t _workspace_size`: 临时存储空间大小(当前实现为 0,无需额外工作空间)
- **Core Methods**:
  - `create(handle, desc_ptr, y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo)`: 静态工厂方法,验证输入张量描述符,提取形状和步长信息,创建并初始化 RoPE 描述符对象。返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream)`: 执行 RoPE 计算的主方法。根据数据类型和位置索引类型分发到特化的模板函数,启动 MetaX GPU 内核
  - `~Descriptor()`: 析构函数,释放不透明内部状态
- **Lifecycle**: 通过 `create()` 静态方法构造,内部持有 `device::metax::Handle::Internal` 的共享指针以管理设备资源,析构时自动清理

### `Descriptor::Opaque`
- **Location**: `rope_metax.maca:38-40`
- **Primary Function**: 封装 MetaX 设备特定的内部状态,与上层接口解耦
- **Key Members**:
  - `std::shared_ptr<device::metax::Handle::Internal> internal`: MetaX 设备句柄的内部表示,提供线程池、内核启动参数等信息
- **Lifecycle**: 由 `Descriptor::create()` 动态分配,`Descriptor` 析构时释放

### `RoPEInfo`
- **Location**: `../rope.h:56-204` (父头文件定义,本模块使用)
- **Primary Function**: 存储 RoPE 操作的静态元数据,在 `create()` 阶段从张量描述符中提取并验证
- **Key Members**:
  - `infiniDtype_t data_type`: 数据类型 (支持 F16, BF16, F32, F64)
  - `infiniDtype_t pos_type`: 位置 ID 类型 (支持所有整数类型)
  - `size_t batch, seqlen, nhead, dhead`: 张量维度 (批量大小、序列长度、注意力头数、每头维度)
  - `size_t table_len, table_dim`: 正弦/余弦表长度和维度 (table_dim = dhead / 2)
  - `ptrdiff_t y_stride_batch, y_stride_seqlen, y_stride_nhead`: 输出张量步长
  - `ptrdiff_t x_stride_batch, x_stride_seqlen, x_stride_nhead`: 输入张量步长
  - `bool has_batch_dim`: 输入/输出是否为 4D 张量 [batch, seqlen, nhead, dhead],否则为 3D [seqlen, nhead, dhead]
  - `bool pos_has_batch_dim`: 位置 ID 是否为 2D 张量 [batch, seqlen],否则为 1D [seqlen]
  - `infiniopRoPEAlgo_t algo`: RoPE 算法变体 (GPT-J 或标准 RoPE)
- **Core Methods**:
  - `createRoPEInfo(...)`: 静态工厂方法,验证张量形状一致性、数据类型兼容性、步长连续性,提取并构造 `RoPEInfo` 对象
- **Design Pattern**: 使用静态工厂方法和 `utils::Result<T>` 返回值实现安全的错误处理

## 3. API Interface

```cpp
namespace op::rope::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    // 创建 RoPE 描述符
    static infiniStatus_t create(
        infiniopHandle_t handle,                    // [in] MetaX 设备句柄
        Descriptor **desc_ptr,                      // [out] 输出描述符指针
        infiniopTensorDescriptor_t y_desc,          // [in] 输出张量描述符 [batch?, seqlen, nhead, dhead]
        infiniopTensorDescriptor_t x_desc,          // [in] 输入张量描述符 [batch?, seqlen, nhead, dhead]
        infiniopTensorDescriptor_t pos_desc,        // [in] 位置 ID 描述符 [batch?, seqlen]
        infiniopTensorDescriptor_t sin_desc,        // [in] 正弦表描述符 [table_len, table_dim]
        infiniopTensorDescriptor_t cos_desc,        // [in] 余弦表描述符 [table_len, table_dim]
        infiniopRoPEAlgo_t algo                     // [in] RoPE 算法 (INFINIOP_ROPE_ALGO_DEFAULT 或 INFINIOP_ROPE_ALGO_GPT_J)
    );

    // 执行 RoPE 计算
    infiniStatus_t calculate(
        void *workspace,                            // [in] 工作空间指针 (当前未使用)
        size_t workspace_size,                      // [in] 工作空间大小
        void *y,                                    // [out] 输出张量设备指针
        const void *x,                              // [in] 输入张量设备指针
        const void *pos_ids,                        // [in] 位置 ID 设备指针
        const void *sin_table,                      // [in] 正弦表设备指针
        const void *cos_table,                      // [in] 余弦表设备指针
        void *stream                                // [in] MetaX 计算流 (hcStream_t)
    ) const;

    size_t workspaceSize() const;                  // 返回工作空间大小 (当前为 0)
    ~Descriptor();
};

} // namespace op::rope::metax
```

## 4. Usage Example

```cpp
// 示例: 在 MetaX GPU 上执行 RoPE 位置编码
// 假设输入形状为 [batch=2, seqlen=128, nhead=32, dhead=128]

#include "infiniop/ops/rope/metax/rope_metax.h"

// 1. 初始化 MetaX 设备句柄
infiniopHandle_t handle;
infiniStatus_t status = infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);

// 2. 创建张量描述符
// 输入/输出: [2, 128, 32, 128] float16
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc, INFINI_DTYPE_F16, 4,
                               {2, 128, 32, 128}, {32*128, 128, 1, 0});
infiniopCreateTensorDescriptor(handle, &y_desc, INFINI_DTYPE_F16, 4,
                               {2, 128, 32, 128}, {32*128, 128, 1, 0});

// 位置 ID: [2, 128] int32_t (每批次独立位置)
infiniopTensorDescriptor_t pos_desc;
infiniopCreateTensorDescriptor(handle, &pos_desc, INFINI_DTYPE_I32, 2,
                               {2, 128}, {128, 1});

// 正弦/余弦表: [8192, 64] float16 (预计算的最大序列长度)
infiniopTensorDescriptor_t sin_desc, cos_desc;
infiniopCreateTensorDescriptor(handle, &sin_desc, INFINI_DTYPE_F16, 2,
                               {8192, 64}, {64, 1});
infiniopCreateTensorDescriptor(handle, &cos_desc, INFINI_DTYPE_F16, 2,
                               {8192, 64}, {64, 1});

// 3. 创建 RoPE 描述符
op::rope::metax::Descriptor *rope_desc;
status = op::rope::metax::Descriptor::create(
    handle, &rope_desc, y_desc, x_desc, pos_desc,
    sin_desc, cos_desc, INFINIOP_ROPE_ALGO_DEFAULT
);

// 4. 分配设备内存并初始化数据
half *d_x, *d_y, *d_sin, *d_cos;
int32_t *d_pos;
size_t x_size = 2 * 128 * 32 * 128 * sizeof(half);
hcMalloc((void**)&d_x, x_size);
hcMalloc((void**)&d_y, x_size);
hcMalloc((void**)&d_sin, 8192 * 64 * sizeof(half));
hcMalloc((void**)&d_cos, 8192 * 64 * sizeof(half));
hcMalloc((void**)&d_pos, 2 * 128 * sizeof(int32_t));
// ... (通过 hcMemcpyH2D 填充数据)

// 5. 创建计算流
hcStream_t stream;
hcStreamCreate(&stream);

// 6. 执行 RoPE 计算
status = rope_desc->calculate(
    nullptr, 0,              // 无需工作空间
    d_y,                      // 输出
    d_x,                      // 输入
    d_pos,                    // 位置 ID
    d_sin, d_cos,             // 正弦/余弦表
    stream                    // MetaX 流
);

// 7. 同步并清理
hcStreamSynchronize(stream);
hcFree(d_x); hcFree(d_y); hcFree(d_sin); hcFree(d_cos); hcFree(d_pos);
hcStreamDestroy(stream);
delete rope_desc;
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### **Kernel Architecture**
- **Kernel Source**: 直接复用 CUDA 实现的 `ropeThreadPerItemBlock` 内核 (来自 `../cuda/kernel.cuh`)
- **Kernel Wrapper**: `ropeThreadPerItemKernel` 模板函数封装 CUDA 内核,适配 MetaX 启动语法
- **Grid Configuration**:
  - **3D 张量** `[seqlen, nhead, dhead]`: 使用 2D 网格 `dim3(seqlen, nhead)`, batch 维度固定为 1
  - **4D 张量** `[batch, seqlen, nhead, dhead]`: 使用 3D 网格 `dim3(seqlen, nhead, batch)`
  - 每个线程块处理一个 `(seqlen, nhead)` 位置,线程数 = `max(table_dim, maxThreadsPerBlock)`
- **Thread Mapping**: 每个线程块负责处理一个 token 的单个注意力头,线程并行处理 `table_dim` (即 `dhead/2`) 个旋转角度对

### **Algorithm Details**
- **Standard RoPE** (`IsGPTJ = false`):
  - 输入按位置交错: `[x0, x1, x2, x3, ...]` 其中偶数索引应用 cos,奇数索引应用 sin
  - 旋转逻辑: `y[2*i] = x[2*i]*cos - x[2*i+1]*sin`, `y[2*i+1] = x[2*i]*sin + x[2*i+1]*cos`
  - 内存访问: 位置 `i` 从 sin/cos 表读取 `table_offset + i`,从输入读取 `offset + i` 和 `offset + i + table_dim`
- **GPT-J RoPE** (`IsGPTJ = true`):
  - 输入按连续对组织: `[x0, x1], [x2, x3], ...` 每对独立旋转
  - 旋转逻辑相同,但访问模式为 `offset + 2*i` 和 `offset + 2*i + 1`
  - **SIMD 优化**: 对于 FP16 使用 `half2` 向量化加载/存储,对于 BF16 使用 `bfloat162` 向量化操作

### **Type Dispatch Strategy**
实现两级类型分发以支持数据类型和位置索引类型的任意组合:
1. **外层循环** (`calculate()` 方法): 根据 `data_type` 分发 FP16/BF16/FP32/FP64
2. **内层循环** (`ROPE_TYPE` 宏): 根据 `pos_type` 分发 U8/U16/U32/U64/I8/I16/I32/I64
3. **模板实例化**: 最终生成 `4 × 8 = 32` 个特化版本的 `calculateRoPE<Tdata, Tindex>` 模板函数

### **Memory Access Patterns**
- **输入/输出**: 支持非连续步长,通过 `y_stride_*` 和 `x_stride_*` 计算 `y_offset` 和 `x_offset`
  - 公式: `offset = batch*stride_batch + seqlen*stride_seqlen + head*stride_nhead`
  - 要求最后一维 (dhead) 步长必须为 1 (连续内存)
- **位置 ID**:
  - 1D 张量 `[seqlen]`: 所有批次共享位置表,`pos_offset = seq_idx`
  - 2D 张量 `[batch, seqlen]`: 每批次独立位置表,`pos_offset = batch*pos_stride_batch + seq_idx`
- **正弦/余弦表**: 要求完全连续内存,`table_offset = pos_id * table_dim`

### **MetaX-Specific Optimizations**
- **Kernel Launch**: 使用 `INFINIOP_METAX_KERNEL` 宏 (展开为 `__global__ void`) 兼容 MetaX HIP/CUDA 编译器
- **Device Properties**: 从 `device::metax::Handle::Internal` 查询 `maxThreadsPerBlock()` 动态调整线程块大小
- **Block Size**: `nthreads = max(int(table_dim), maxThreadsPerBlock)`,确保足够的线程并行度
- **Type Aliases**: 使用 `cuda_bfloat16 = hpcc_bfloat16` 等 typedef 桥接 CUDA 和 MetaX 类型系统

### **Concurrency & Stream Management**
- **异步执行**: `calculate()` 方法立即返回,实际计算在 `hcStream_t` 流上异步执行
- **Stream Safety**: 同一流上的连续内核调用自动排序,无需显式同步
- **Resource Sharing**: 通过 `device::metax::Pool` 复用 `hcblasHandle_t` 和 `hcdnnHandle_t`,避免频繁创建/销毁开销

### **Error Handling**
- **Descriptor Creation**: 使用 `CHECK_RESULT(info)` 宏检查 `RoPEInfo::createRoPEInfo()` 返回的 `Result<T>`,失败时返回错误码
- **Type Dispatch**: 不支持的类型组合返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **Kernel Launch**: CUDA/HIP 启动失败时,错误通过 `CHECK_METAX(API)` 宏传播

### **Dependencies**
- **External Libraries**:
  - `hcblas` / `mcblas`: MetaX BLAS 库 (通过 `device::metax::Handle` 间接使用)
  - `hcdnn` / `mcdnn`: MetaX DNN 库 (通过 `device::metax::Handle` 间接使用)
  - `hpcc_fp8.h` / `maca_fp8.h`: MetaX FP8 类型定义 (编译时选择)
- **Internal Modules**:
  - `../rope.h`: 提供 `RoPEInfo` 和 `DESCRIPTOR` 宏定义
  - `../cuda/kernel.cuh`: 提供 `ropeThreadPerItemBlock` 设备函数
  - `../../devices/metax/metax_common.h`: 提供 `device::metax::Handle` 和工具宏
  - `../../devices/metax/metax_kernel_common.h`: 提供 MetaX 内核定义和类型别名
- **Design Patterns**:
  - **Pimpl Idiom**: `Descriptor::Opaque` 隐藏设备特定实现细节
  - **Template Metaprogramming**: 编译期类型分发和算法选择 (`IsGPTJ` 模板参数)
  - **RAII**: 资源管理通过智能指针 (`std::shared_ptr`) 和析构函数自动处理
  - **Static Factory**: `create()` 方法封装复杂的构造逻辑和验证
