# Metax 后端逐元素加法操作实现文档

本模块实现了 Moore Threads 元语后端（Metax）的逐元素加法操作，支持多种数据类型（FP16、FP32、FP64、BF16、INT32、INT64），基于统一的逐元素操作框架实现张量加法运算。

## 1. 模块结构

- **`add_metax.h`**: 定义 Metax 加法操作的描述符类接口，继承自通用逐元素操作描述符
- **`add_metax.maca`**: 实现 Metax 加法操作的核心逻辑，包括描述符创建与计算调度

## 2. 核心类

### `op::add::metax::Descriptor`
- **位置**: `add_metax.h` / `add_metax.maca`
- **主要功能**: Metax 设备上的加法操作描述符，管理操作的生命周期与执行
- **继承关系**: 通过宏 `ELEMENTWISE_DESCRIPTOR(add, metax)` 从基类继承，自动获得标准成员变量与接口
- **关键成员**（继承自基类）:
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型
  - `_info`: `op::elementwise::ElementwiseInfo`，存储张量形状、步长、广播等元信息
  - `_device_info`: `op::elementwise::metax::DeviceImpl *`，Metax 设备实现对象
  - `_workspace_size`: `size_t`，所需工作空间大小（存储设备端元数据）
  - `_device`, `_device_id`: 设备属性标识

#### 核心方法

**`create(handle_, desc_ptr, out_desc, input_desc_vec)`**
- **功能**: 创建加法操作描述符实例，验证输入输出张量兼容性
- **算法流程**:
  1. 转换句柄为 Metax 设备句柄类型
  2. 提取输入张量 A、B 与输出张量 C 的描述符信息
  3. 验证数据类型支持（F16/F32/F64/BF16/I32/I64）
  4. 验证三个张量形状完全一致（`CHECK_SAME_SHAPE`）
  5. 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏：
     - 创建 `ElementwiseInfo` 对象（计算布局、广播信息）
     - 计算 workspace 大小 = 元数据大小 + 输入指针数组大小
     - 创建 `DeviceImpl` 对象（封装设备端实现）
     - 构造并返回 `Descriptor` 实例
- **时间复杂度**: O(1)（仅做验证与对象构造）

**`calculate(workspace, workspace_size, output, inputs, stream)`**
- **功能**: 在 Metax 设备上执行加法计算 kernel
- **算法流程**:
  1. 检查 workspace 大小是否满足需求
  2. 根据数据类型分发至对应的模板实例化：
     - 调用 `_device_info->calculate<256, cuda::AddOp, Ttype>()`
     - `BLOCK_SIZE=256`：每个 CUDA block 使用 256 个线程
     - `cuda::AddOp`：加法运算符（复用 CUDA 定义）
     - `Ttype`：具体类型（half/float/double 等）
- **Kernel 实现**:
  - 位于 `elementwise_metax.maca` 的通用 `elementwiseKernel` 模板
  - 每个 CUDA 线程处理一个输出元素
  - 支持非连续张量、广播语义
- **空间复杂度**: 需要 O(N) workspace，N 为张量维度数乘输入数量

**`~Descriptor()`**
- **功能**: 默认析构函数，释放基类管理的资源

## 3. API 接口

```cpp
namespace op::add::metax {

class Descriptor {
public:
    ~Descriptor();

    // 创建加法操作描述符
    static infiniStatus_t create(
        infiniopHandle_t handle_,              // [输入] Metax 设备句柄
        Descriptor **desc_ptr,                 // [输出] 描述符指针
        infiniopTensorDescriptor_t out_desc,   // [输入] 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [输入] 输入张量描述符向量 [A, B]
    );

    // 执行加法计算
    infiniStatus_t calculate(
        void *workspace,                       // [输入] 设备端工作空间
        size_t workspace_size,                 // [输入] 工作空间大小
        void *output,                          // [输出] 输出张量数据
        std::vector<const void *> inputs,      // [输入] 输入张量数据向量 [A, B]
        void *stream                           // [输入] Metax 流
    ) const;
};

}
```

## 4. 使用示例

```cpp
// 示例：在 Metax 设备上执行张量加法 C = A + B

#include "add_metax.h"
#include "infiniop.h"

// 1. 创建设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, device_type_metax, device_id);

// 2. 创建张量描述符（假设形状为 [1024, 1024]）
int64_t shape[2] = {1024, 1024};
int64_t strides[2] = {1024, 1};

infiniopTensorDescriptor_t desc_a, desc_b, desc_c;
infiniopCreateTensorDescriptor(&desc_a, INFINI_DTYPE_F16, 2, shape, strides);
infiniopCreateTensorDescriptor(&desc_b, INFINI_DTYPE_F16, 2, shape, strides);
infiniopCreateTensorDescriptor(&desc_c, INFINI_DTYPE_F16, 2, shape, strides);

// 3. 创建加法操作描述符
op::add::metax::Descriptor *add_desc;
std::vector<infiniopTensorDescriptor_t> inputs = {desc_a, desc_b};
auto status = op::add::metax::Descriptor::create(handle, &add_desc, desc_c, inputs);

// 4. 分配设备内存并输入数据
half *d_a, *d_b, *d_c;
size_t bytes = 1024 * 1024 * sizeof(half);
hcMalloc((void**)&d_a, bytes);
hcMalloc((void**)&d_b, bytes);
hcMalloc((void**)&d_c, bytes);
// ... 从主机拷贝数据到 d_a, d_b

// 5. 分配 workspace（由 Descriptor::create 计算大小）
void *workspace;
hcMalloc(&workspace, add_desc->_workspace_size);

// 6. 创建执行流
hcStream_t stream;
hcStreamCreate(&stream);

// 7. 执行加法计算
std::vector<const void *> input_data = {d_a, d_b};
status = add_desc->calculate(workspace, add_desc->_workspace_size, d_c, input_data, stream);

// 8. 同步并获取结果
hcStreamSynchronize(stream);
// ... 从 d_c 拷贝结果回主机

// 9. 清理资源
hcFree(d_a); hcFree(d_b); hcFree(d_c); hcFree(workspace);
hcStreamDestroy(stream);
delete add_desc;
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **Workspace 策略**: 使用单一连续 workspace 存储设备端元数据：
  - 输入指针数组（`void *inputs[N]`）：N 个输入指针
  - 元数据区域（从 `ElementwiseInfo` 提取）：
    - 输出形状（`size_t[ndim]`）
    - 输出步长（`ptrdiff_t[ndim]`）
    - N 个输入形状（`size_t[N * ndim]`）
    - N 个输入步长（`ptrdiff_t[N * ndim]`）
    - N 个输入连续标志（`bool[N]`）
    - N 个输入广播标志（`bool[N]`）
- **传输优化**: 通过 `hcMemcpyAsync` 一次性拷贝所有元数据到设备，减少 Host-Device 交互次数

### 并发执行
- **Kernel 并行模型**:
  - 使用 2D Grid-Block 结构：`gridIdx * blockDim.x + threadIdx.x`
  - BLOCK_SIZE 固定为 256，根据设备属性动态调整实际线程数
  - Grid 大小限制为设备最大 GridDimX（通常 65535 或更高）
  - 超大张量通过循环多次启动 kernel 处理（step = gridDims.x * blockDims.x）
- **流式执行**: 支持异步执行，通过 `hcStream_t` 管理并发操作
- **线程安全**: 多个流可并发执行同一 descriptor 的计算（只读访问元数据）

### 性能优化
- **硬件指令优化**（通过复用 `cuda::AddOp`）:
  - FP16：使用 `__hadd` 内置指令
  - FP2x16：使用 `__hadd2` 向量化指令
  - FP32：使用 `__fadd_rd` 舍入控制加法
  - 其他类型：直接使用 `+` 运算符
- **连续内存优化**: 对连续张量直接使用线性索引，避免多维索引计算开销
- **分支消除**: 通过模板特化在编译期解析数据类型，避免运行期分支
- **网格步进策略**: 单次 kernel 启动处理 `gridDims.x * blockDims.x` 元素，减少 kernel 启动开销

### 错误处理
- **类型验证**: 编译期检查 `Op::num_inputs == 2`，运行期检查数据类型支持列表
- **形状验证**: 运行期验证输入输出形状一致性（`CHECK_SAME_SHAPE`）
- **资源检查**: 验证 workspace 大小不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **错误传播**: 所有底层 HC API 调用通过 `CHECK_METAX` 宏传播错误状态
- **空张量处理**: `output_size == 0` 时直接返回成功，避免无效 kernel 启动

### 依赖关系
- **上级依赖**:
  - `op::elementwise::metax::ElementwiseInfo`：张量元数据管理与索引计算
  - `op::elementwise::metax::DeviceImpl`：Metax 设备端 kernel 启动与管理
  - `device::metax::Handle`：Moore Threads 设备句柄与属性
  - `op::add::cuda::AddOp`：加法运算符定义（复用 CUDA 实现）
- **外部依赖**:
  - Moore Threads HC Runtime (`hcMemcpyAsync`, `hcMalloc`, `hcStreamCreate`)
  - CUDA 互操作层（支持 CUDA 类型与内建函数）
  - InfiniOP 基础设施（`infiniopHandle_t`, `infiniopTensorDescriptor_t`）

### 设计模式
- **策略模式**: 通过模板参数 `Op` 将加法逻辑抽象为可配置策略
- **工厂模式**: `Descriptor::create` 静态方法作为对象构造工厂
- **桥接模式**: `Descriptor`（接口）通过 `DeviceImpl`（实现）解耦高层逻辑与设备细节
- **模板方法模式**: `elementwiseKernel` 定义算法骨架，`Op` 定义具体操作
- **RAII**: 使用 `std::shared_ptr` 管理 `DeviceImpl::Opaque` 生命周期
