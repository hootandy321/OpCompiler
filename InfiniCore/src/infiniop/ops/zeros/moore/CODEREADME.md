# Moore 后端 Zeros 操作核心实现文档

本模块实现了 Infini 框架中 Zeros（零值填充）操作的 Moore 硬件后端，支持 15 种数据类型的张量零值初始化。通过复用 elementwise 操作的基础设施，实现高效的 GPU 并行零值写入。

## 1. 模块结构

- **`zeros_moore.h`**: 操作描述符的 API 声明，通过 `ELEMENTWISE_DESCRIPTOR` 宏定义操作接口
- **`zeros_moore_kernel.h`**: CUDA 设备端内核算子定义，实现类型擦除的零值生成函数对象
- **`zeros_moore.mu`**: Moore 后端的实现主文件，包含操作描述符的创建、计算调度逻辑

## 2. 核心类

### `ZerosOp` (CUDA Kernel Functor)
- **位置**: `zeros_moore_kernel.h:6-45`
- **主要功能**: 设备端函数对象，为 15 种数据类型提供编译期类型安全的零值生成
- **关键特性**:
  - 静态成员 `num_inputs = 1` 标识单输入操作（输入形状用于广播匹配）
  - 使用 `if constexpr` 实现编译期分支，无运行时开销
  - 支持浮点数（FP8/F16/F32/F64/BF16）、整数（I8/I16/I32/I64）、无符号整数（U8/U16/U32/U64）、布尔类型
- **核心方法**:
  ```cpp
  template <typename T>
  __device__ __forceinline__ T operator()(const T &x) const;
  ```
  - **语义**: 忽略输入参数 `x`，返回类型 `T` 的零值表示
  - **类型映射**:
    - 布尔类型 → `false`
    - 整数/无符号整数 → `0`
    - `cuda_fp8_e4m3` → `cuda_fp8_e4m3(0.0f)` (通过浮点构造)
    - `half` → `__float2half(0.0f)` (CUDA 内置转换)
    - `float` → `0.0f`
    - `double` → `0.0`
    - `cuda_bfloat16` → `__float2bfloat16(0.0f)` (CUDA 内置转换)
  - **复杂度**: O(1) 编译期优化

### `Descriptor` (操作描述符)
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(zeros, moore)` 宏实例化于 `zeros_moore.h:6`
- **主要功能**: 管理 Zeros 操作的生命周期、元数据验证、工作空间计算、内核调度
- **继承体系**: 继承自 `InfiniopDescriptor`（基础操作接口）
- **核心成员**:
  - `_dtype: infiniDtype_t`: 输出张量的数据类型（15 种支持类型之一）
  - `_info: ElementwiseInfo`: 封装输入/输出张量的形状、步幅、连续性等元数据
  - `_device_info: unique_ptr<op::elementwise::moore::DeviceImpl>`: Moore 设备实现指针（类型擦除的内核调度器）
  - `_workspace_size: size_t`: 设备端工作空间大小（存储元数据 + 输入指针数组）
- **生命周期**:
  1. **创建**: `Descriptor::create()` 静态工厂方法
  2. **配置**: 构造时初始化 dtype、元数据、设备信息、工作空间
  3. **执行**: `calculate()` 方法派发内核
  4. **销毁**: `~Descriptor()` 默认析构（RAII 管理智能指针）

## 3. API 接口

```cpp
namespace op::zeros::moore {

// 操作描述符（通过 ELEMENTWISE_DESCRIPTOR 宏生成）
class Descriptor final : public InfiniopDescriptor {
public:
    // 析构函数
    ~Descriptor();

    // 获取工作空间大小
    size_t workspaceSize() const;

    // 创建操作描述符
    static infiniStatus_t create(
        infiniopHandle_t handle_,                 // [in] Moore 设备句柄
        Descriptor **desc_ptr,                    // [out] 输出的描述符指针
        infiniopTensorDescriptor_t out_desc,      // [in] 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量（用于形状匹配）
    );

    // 执行零值填充计算
    infiniStatus_t calculate(
        void *workspace,                          // [in] 设备端工作空间指针
        size_t workspace_size,                    // [in] 工作空间大小（需 >= workspaceSize()）
        void *output,                             // [out] 输出张量设备指针
        std::vector<const void *> inputs,         // [in] 输入张量设备指针数组（实际未使用）
        void *stream                              // [in] MUSA/CUDA 流
    ) const;
};

} // namespace op::zeros::moore
```

### 关键返回值
- `INFINI_STATUS_SUCCESS`: 操作成功
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型（非 15 种类型之一）
- `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间不足
- `INFINI_STATUS_NOT_IMPLEMENTED`: 复数类型暂不支持（C16/C32/C64/C128）

## 4. 使用示例

```cpp
// 上下文：使用 Moore 后端初始化一个 float32 类型张量为零
#include "infiniop/ops/zeros/moore/zeros_moore.h"

// 1. 准备张量描述符
std::vector<size_t> shape = {1024, 1024};
auto output_desc = infiniopTensorDescriptor_t(...); // 创建 float32 输出描述符
auto input_desc = infiniopTensorDescriptor_t(...);  // 创建输入描述符（形状需匹配）

// 2. 创建 Zeros 操作描述符
op::zeros::moore::Descriptor* zeros_desc = nullptr;
infiniStatus_t status = op::zeros::moore::Descriptor::create(
    moore_handle,           // Moore 设备句柄
    &zeros_desc,            // 输出描述符指针
    output_desc,            // 输出张量描述符
    {input_desc}            // 输入描述符向量（仅用于形状验证）
);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配工作空间和输出内存
size_t workspace_size = zeros_desc->workspaceSize();
void* d_workspace = nullptr;
void* d_output = nullptr;
musaMalloc(&d_workspace, workspace_size);
musaMalloc(&d_output, 1024 * 1024 * sizeof(float));

// 4. 执行零值填充计算
musaStream_t stream;
musaStreamCreate(&stream);
status = zeros_desc->calculate(
    d_workspace,    // 设备端工作空间
    workspace_size, // 工作空间大小
    d_output,       // 输出张量设备指针
    {nullptr},      // 输入指针（未使用，但需提供一个元素）
    stream          // MUSA 流
);

// 5. 同步与清理
musaStreamSynchronize(stream);
musaFree(d_workspace);
musaFree(d_output);
delete zeros_desc;
musaStreamDestroy(stream);
```

## 5. 实现细节

### 设计模式
- **策略模式 (Strategy Pattern)**: `ZerosOp` 作为可调用策略，由 elementwise 框架统一调度
- **工厂模式 (Factory Pattern)**: `Descriptor::create()` 静态工厂方法封装对象构造
- **模板方法模式 (Template Method Pattern)**: `ELEMENTWISE_DESCRIPTOR` 宏生成标准化描述符结构
- **RAII**: 使用 `unique_ptr` 管理 `DeviceImpl` 生命周期

### 内存管理
- **工作空间布局**:
  ```
  [输入指针数组 (N * sizeof(void*))] [元数据块 (ElementwiseInfo)]
  ```
  - 输入指针数组: 虽然输入值未使用，但框架需传递指针以保持统一接口
  - 元数据块: 包含 ndim、形状、步幅、连续性标记，由 `ElementwiseInfo` 管理
- **零拷贝优化**: 对于连续张量，内核直接使用线性索引；对于非连续张量，运行时计算 `indexToOffset`
- **内存分配**: 调用方负责分配工作空间和输出内存，描述符仅管理元数据

### 并发模型
- **内核配置**:
  - `BLOCK_SIZE = 256`: 每个 CUDA 线程块 256 个线程（固定值，平衡寄存器占用和并行度）
  - Grid 大小: `min(ceil(output_size / 256), device.gridSizeX())`
  - 步进循环: 对于超大张量（> grid * block），通过 `offset` 参数多次启动内核
- **线程安全**: 每个线程处理独立输出元素，无竞争条件
- **流并发**: 支持异步执行，调用方可通过多个流并发提交不同操作

### 性能优化
- **编译期优化**:
  - `if constexpr` 完全消除类型分支（零运行时开销）
  - `__forceinline__` 强制内联函数对象调用
  - 模板特化为每种数据类型生成专用内核代码
- **内存访问优化**:
  - 连续张量：合并访问（coalesced access），最大化带宽利用率
  - 非连续张量：`InputIndexer` 封装索引计算，避免重复逻辑
- **类型安全**: 编译期类型检查防止类型混淆错误

### 错误处理
- **创建阶段验证**:
  - `CHECK_DTYPE`: 确保输出类型是 15 种支持类型之一
  - `CHECK_SAME_SHAPE`: 验证输入/输出形状完全匹配（用于广播一致性）
- **执行阶段验证**:
  - 工作空间大小检查：`workspace_size < _workspace_size` 返回 `INSUFFICIENT_WORKSPACE`
  - 复数类型返回 `NOT_IMPLEMENTED`（C16/C32/C64/C128）
  - 未知类型返回 `BAD_TENSOR_DTYPE`
- **CUDA/MUSA 错误传播**: `CHECK_MOORE` 宏将设备 API 错误转换为 `infiniStatus_t`

### 依赖关系
- **上游依赖**:
  - `../../../elementwise/moore/elementwise_moore_api.h`: Moore 后端 elementwise 基础设施
  - `../../../elementwise/moore/elementwise_moore.h`: 内核启动器和 `DeviceImpl` 实现
  - `../cuda/kernel.cuh`: CUDA 后端内核声明（复用 Moore 实现路径）
- **类型依赖**:
  - CUDA 类型: `cuda_fp8_e4m3`, `half`, `cuda_bfloat16`（需 CUDA 工具包）
  - MUSA 类型: 摩尔线程架构（`musaStream_t`, `musaMemcpyAsync`）
- **框架集成**:
  - `InfiniopDescriptor`: 基础操作接口
  - `ElementwiseInfo`: 元数据容器（形状、步幅、连续性）
  - `device::moore::Handle`: Moore 设备句柄（提供设备属性查询）

### 数据流
```
用户 API (Descriptor::create)
    ↓
验证 dtype/shape → 生成 ElementwiseInfo
    ↓
分配工作空间 → 创建 DeviceImpl
    ↓
用户 API (Descriptor::calculate)
    ↓
切换 dtype → 调用 DeviceImpl::calculate<T>
    ↓
复制元数据到设备 → 启动 elementwiseKernel<<<grid, block>>>
    ↓
每个线程执行 ZerosOp::operator() → 写入零值到 output
```

### 关键算法
- **零值生成算法**:
  - 时间复杂度: O(n)，其中 n 为输出张量元素数量
  - 空间复杂度: O(1) 额外空间（仅工作空间，与 n 无关）
  - 并行度: O(n)（每个元素独立处理）
- **索引计算** (`device::moore::indexToOffset`):
  - 输入: 线性索引 `idx`，维度数 `ndim`，形状 `shape[]`，步幅 `strides[]`
  - 输出: 多维偏移量
  - 复杂度: O(ndim)，通常 ndim ≤ 8，开销可忽略

### 扩展性
- **添加新数据类型**: 在 `ZerosOp::operator()` 中添加新的 `if constexpr` 分支
- **调整块大小**: 修改 `calculate()` 中的模板参数 `256`（需权衡寄存器使用和占用率）
- **支持复数类型**: 实现复数零值（`0+0i`），移除 `NOT_IMPLEMENTED` 分支

### 限制与已知问题
- **输入张量未使用**: 输入值被忽略，但需传递一个输入描述符用于形状匹配（接口设计约束）
- **工作空间开销**: 即使对于连续张量，也需复制元数据到设备（可优化为路径分离）
- **类型代码重复**: `calculate()` 中有 14 个 case 分支（可通过模板宏简化）
