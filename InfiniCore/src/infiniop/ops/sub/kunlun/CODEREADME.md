# Kunlun GPU 减法运算 (Subtraction Operation) 核心实现文档

本模块实现了昆仑（KUNLUN）XPU 设备上的张量逐元素减法运算。通过复用通用逐元素运算框架，提供了高效的 GPU 并行减法计算能力，支持 FP16、FP32 和 BF16 三种浮点数据类型，具备广播机制和非连续张量的完整支持。

## 1. 模块结构

- **`kernel.h`**: 定义减法运算的设备端核心算子 `SubOp`，实现逐元素减法逻辑
- **`sub_kunlun.h`**: 减法运算的 API 声明，通过宏 `ELEMENTWISE_DESCRIPTOR` 生成完整的 Descriptor 类接口
- **`sub_kunlun.xpu`**: 减法运算的实现文件，包含算子描述符的创建和计算调度逻辑

## 2. 核心类

### `SubOp` (减法算子)
- **位置**: `kernel.h`
- **主要功能**: 设备端函数对象，在 XPU 核函数中执行单个元素或小向量的减法运算
- **关键成员**:
  - `num_inputs`: 静态常量，固定值为 2，表示减法运算需要两个输入操作数
- **核心方法**:
  - `operator()(const T *inputs) const`: 通用模板版本，从输入数组中读取两个操作数 `a = inputs[0]` 和 `b = inputs[1]`，返回 `a - b`。支持 FP32、FP16 等类型的直接运算
  - `operator()(const bfloat16_t *inputs) const`: BF16 特化版本，先将两个 BF16 输入转换为 float 类型（`__bfloat162float`），执行 float 减法后再转回 BF16（`__float2bfloat16`），确保计算精度
- **生命周期**: 无状态函数对象，编译期静态常量，无构造/析构逻辑

### `Descriptor` (减法描述符)
- **位置**: 由 `ELEMENTWISE_DESCRIPTOR(sub, kunlun)` 宏在 `sub_kunlun.h` 中生成，实现在 `sub_kunlun.xpu`
- **主要功能**: 封装减法运算的元数据、设备信息和执行接口，负责参数验证、工作空间计算和内核调度
- **关键成员**:
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型（FP16/FP32/BF16）
  - `_info`: `op::elementwise::ElementwiseInfo`，存储输入输出张量的形状、步长、连续性和广播信息的元数据结构
  - `_device_info`: `std::unique_ptr<op::elementwise::kunlun::DeviceImpl>`，昆仑 XPU 设备实现层的封装对象，管理底层内核启动逻辑
  - `_workspace_size`: `size_t`，设备端工作空间大小，用于存储元数据和输入指针数组
- **核心方法**:
  - `create(...)`: 静态工厂方法，验证输入输出张量描述符的形状一致性（`CHECK_SAME_SHAPE`）和数据类型合法性（仅支持 FP16/F32/BF16），构造 `ElementwiseInfo` 元数据，创建 `DeviceImpl` 实例并计算工作空间大小
  - `calculate(...)`: 执行减法运算的主入口，检查工作空间是否充足，根据 `_dtype` 分发到对应的模板实例化（`_device_info->calculate<8, SubOp, T>`），其中模板参数 `8` 为 XPU 线程块大小
- **生命周期**: 由 `create` 工厂方法动态分配，用户负责释放（析构函数默认实现）

## 3. API 接口

```cpp
namespace op::sub::kunlun {

// 设备端减法算子
typedef struct SubOp {
    static constexpr int num_inputs = 2;

    // 通用版本：支持 float, half 等
    template <typename T>
    inline __device__ T operator()(const T *inputs) const;

    // BF16 特化版本：提升精度到 float 计算
    inline __device__ bfloat16_t operator()(const bfloat16_t *inputs) const;
} SubOp;

// 运算描述符（由宏生成）
class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // 获取所需工作空间大小
    size_t workspaceSize() const;

    // 创建描述符并验证参数
    static infiniStatus_t create(
        infiniopHandle_t handle,                    // 昆仑 XPU 句柄
        Descriptor **desc_ptr,                      // 输出：描述符指针
        infiniopTensorDescriptor_t output_desc,     // 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_descs  // 输入张量描述符（需2个）
    );

    // 执行减法运算
    infiniStatus_t calculate(
        void *workspace,                // 设备端工作空间指针
        size_t workspace_size,          // 工作空间大小
        void *output,                   // 输出张量设备指针
        std::vector<const void *> inputs,  // 输入张量设备指针数组（需2个）
        void *stream                    // XPU 计算流
    ) const;
};

} // namespace op::sub::kunlun
```

## 4. 使用示例

```cpp
// 示例：在昆仑 XPU 上执行张量减法 C = A - B

#include "sub_kunlun.h"

// 1. 创建句柄和描述符
infiniopHandle_t handle;
infiniopCreateHandle(KUNLUN, 0, &handle);

// 假设张量形状为 {1024, 1024}，数据类型为 FP32
infiniopTensorDescriptor_t desc_A, desc_B, desc_C;
infiniopCreateTensorDescriptor(handle, &desc_A, INFINI_DTYPE_F32, {1024, 1024});
infiniopCreateTensorDescriptor(handle, &desc_B, INFINI_DTYPE_F32, {1024, 1024});
infiniopCreateTensorDescriptor(handle, &desc_C, INFINI_DTYPE_F32, {1024, 1024});

// 2. 创建减法描述符
op::sub::kunlun::Descriptor *sub_desc = nullptr;
auto status = op::sub::kunlun::Descriptor::create(
    handle,
    &sub_desc,
    desc_C,
    {desc_A, desc_B}
);

// 3. 分配并获取工作空间大小
size_t workspace_size = sub_desc->workspaceSize();
void *workspace_d = nullptr;
xpu_malloc(&workspace_d, workspace_size);

// 4. 分配输入输出张量设备内存
float *A_d, *B_d, *C_d;
size_t tensor_size = 1024 * 1024 * sizeof(float);
xpu_malloc((void**)&A_d, tensor_size);
xpu_malloc((void**)&B_d, tensor_size);
xpu_malloc((void**)&C_d, tensor_size);

// 5. 从主机拷贝数据到设备（假设 A_h, B_h 为主机端数据）
xpu_memcpy_async(A_d, A_h, tensor_size, XPU_HOST_TO_DEVICE, stream);
xpu_memcpy_async(B_d, B_h, tensor_size, XPU_HOST_TO_DEVICE, stream);

// 6. 执行减法运算
status = sub_desc->calculate(
    workspace_d,
    workspace_size,
    C_d,
    {A_d, B_d},
    stream
);

// 7. 将结果拷回主机
float *C_h = new float[1024 * 1024];
xpu_memcpy_async(C_h, C_d, tensor_size, XPU_DEVICE_TO_HOST, stream);
xpu_stream_synchronize(stream);

// 8. 清理资源
delete[] C_h;
xpu_free(A_d); xpu_free(B_d); xpu_free(C_d); xpu_free(workspace_d);
delete sub_desc;
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 算法与性能
- **并行策略**: 复用逐元素运算框架的 XPU 并行模式，采用 2D 网格调度（cluster × core），默认块大小为 8 个 cluster，每 cluster 64 个计算核心
- **计算复杂度**: O(n)，其中 n 为输出张量元素总数，每个元素执行一次减法运算
- **内存访问模式**: 支持非连续张量和广播，通过 `InputIndexer` 在运行时计算输入元素的内存偏移，连续张量则直接使用线性索引优化访问
- **向量化**: 内核循环中使用 `BUFF_SIZE=64` 的局部缓存批处理，减少全局内存访问次数

### 内存管理
- **工作空间布局**: 设备端工作空间分为两部分：前段存储输入指针数组（`N * sizeof(void*)`，N=2），后段存储 `ElementwiseInfo` 元数据（形状、步长、连续性标志、广播标志）
- **数据传输**: 使用 `xpu_memcpy_async` 异步拷贝元数据和指针数组到设备，避免同步阻塞
- **本地内存**: 内核使用 XPU 本地内存（`__local__`）缓存输入元数据和中间结果，通过 `GM2LM_ASYNC/LM2GM_ASYNC` 异步搬运并配合 `mfence` 确保内存一致性

### 并发与同步
- **流语义**: 所有操作在用户提供的 XPU stream 上异步执行，支持流内任务级并行
- **集群同步**: 内核结束前调用 `sync_cluster()` 确保同一 cluster 内所有 core 的写入对全局内存可见
- **线程安全**: 描述符创建后不可变（immutable），同一描述符可多流并发调用 `calculate`

### 数据类型处理
- **FP16/FP32**: 直接使用硬件原生指令执行减法，无类型转换开销
- **BF16**: 特化版本通过 `__bfloat162float` 和 `__float2bfloat16` 内置函数转换为 float 计算，避免精度损失，但增加转换开销
- **类型分发**: `calculate` 方法在主机端根据 `_dtype` switch 分发到正确的模板实例，避免分支进入内核

### 广播机制
- **自动检测**: `ElementwiseInfo::create` 在构建时标记每个输入的广播状态（通过 `hasBroadcastDim` 和形状比较）
- **运行时索引**: `InputIndexer` 根据广播标志和原始形状/步长，使用 `indexToOffset` 将线性索引映射到实际内存偏移，广播维度会重复使用同一元素

### 依赖关系
- **逐元素运算框架**: 核心逻辑继承自 `op::elementwise::kunlun::DeviceImpl`，复用其内核启动、元数据管理和工作空间分配功能
- **昆仑设备层**: 依赖 `device::kunlun::Handle` 和 `device::kunlun::kernel` 命名空间提供的设备管理、内存分配和 XPU 内核工具宏（如 `GM2LM_ASYNC`、`mfence`）
- **张量抽象**: 使用 `infiniopTensorDescriptor_t` 统一张量描述接口，支持跨硬件后端的兼容性

### 设计模式
- **模板方法模式**: `SubOp` 作为策略对象传递给通用 `elementwiseKernel`，实现算法定义与执行框架解耦
- **工厂模式**: `Descriptor::create` 静态方法封装对象构造和参数验证逻辑
- **宏生成**: `ELEMENTWISE_DESCRIPTOR` 宏通过代码生成减少重复代码，统一逐元素运算的描述符接口

### 错误处理
- **参数验证**: 创建阶段检查数据类型（仅支持 FP16/F32/BF16）、形状一致性（输出与所有输入形状相同）
- **工作空间检查**: 运行时验证 `workspace_size >= _workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **状态传播**: 使用 `CHECK_RESULT`、`CHECK_KUNLUN`、`CHECK_STATUS` 宏统一处理错误码，确保异常安全
