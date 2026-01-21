# Ones 算子核心实现文档

`Ones` 算子是 InfiniCore 框架中的基础张量初始化操作，用于创建或填充全1张量。该算子提供跨硬件平台（CPU、NVIDIA GPU、华为昇腾、寒武纪等）的统一接口，通过设备分发器调用对应平台的 kernel 实现。

## 1. 模块结构

- **`include/infinicore/ops/ones.hpp`**: 定义 Ones 算子的公共接口，包括 Ones 类、函数签名和类型定义
- **`src/infinicore/ops/ones/ones.cc`**: 实现 Ones 算子的分发器管理和执行逻辑（当前为空实现）

## 2. 核心类

### `Ones`
- **位置**: `include/infinicore/ops/ones.hpp`
- **主要功能**: 提供全1张量创建和填充操作的核心接口，支持多种硬件后端
- **关键成员**:
  - `schema`: 函数指针类型定义，签名为 `void (*)(Tensor)`，表示接受单个输出张量的操作
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 分发器引用，用于管理不同设备类型的 kernel 注册与查找
- **核心方法**:
  - `execute(Tensor output)`: 算子执行入口，当前为空实现，未来应包含设备设置和分发器调用逻辑
    - 预期流程：设置当前设备 → 通过分发器查找对应 kernel → 执行填充操作
    - 时间复杂度：O(n)，n 为张量元素总数
    - 空间复杂度：O(1) 额外空间
  - `dispatcher()`: 返回全局唯一的静态分发器实例（ Meyer's Singleton 模式）
    - 使用静态局部变量确保线程安全的延迟初始化
    - 分发器维护一个大小为 `Device::Type::COUNT` 的函数指针数组
- **生命周期**:
  - 单例模式：分发器在首次调用 `dispatcher()` 时构造，进程生命周期内持久存在
  - 无需显式析构，由 C++ 运行时在进程退出时自动清理

## 3. API 接口

```cpp
namespace infinicore::op {

// Ones 算子核心类
class Ones {
public:
    // 操作签名：接受单个输出张量，就地填充为全1
    using schema = void (*)(Tensor);

    // 执行全1填充操作（当前为空实现）
    static void execute(Tensor output);

    // 获取全局分发器实例
    static common::OpDispatcher<schema> &dispatcher();
};

// Out-of-place 模式：创建新的全1张量
// 返回值：新分配的张量，所有元素初始化为1
Tensor ones();

// In-place 模式：将现有张量填充为全1
// 参数 output：目标张量，将被就地修改
void ones_(Tensor output);

} // namespace infinicore::op
```

**接口说明**:
- `ones()`: 无参数版本（当前实现缺失参数），应接受形状、数据类型、设备参数创建新张量
- `ones_(Tensor output)`: 就地填充版本，直接修改传入的张量数据
- 两种模式对应不同的使用场景：内存复用优先选择 in-place，代码简洁优先选择 out-of-place

## 4. 使用示例

```cpp
#include "infinicore/ops/ones.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;
using namespace infinicore::op;

// 示例 1: 创建全1张量（预期接口）
void example_create_ones() {
    // 假设接口支持形状和数据类型参数
    Shape shape = {3, 4};
    DataType dtype = DataType::F32;
    Device device = Device::Type::NVIDIA;

    // 创建新的全1张量
    Tensor ones_tensor = ones(shape, dtype, device);
    // ones_tensor 现在包含 3x4 的 float32 全1矩阵
}

// 示例 2: 就地填充为全1（当前可用接口）
void example_inplace_ones() {
    // 创建空张量
    Shape shape = {2, 3, 4};
    Tensor empty = Tensor::empty(shape, DataType::F32, Device::cpu());

    // 就地填充为全1
    ones_(empty);
    // empty 的所有元素现在都是 1.0f
}

// 示例 3: 设备间分发
void example_dispatcher() {
    // 为 NVIDIA GPU 注册 kernel
    Ones::dispatcher().registerDevice(
        Device::Type::NVIDIA,
        [](Tensor output) {
            // GPU kernel 实现逻辑
            // 调用 CUDA 核函数填充为1
        },
        false  // 不覆盖已存在的实现
    );

    // 为 CPU 注册 kernel
    Ones::dispatcher().registerDevice(
        Device::Type::CPU,
        [](Tensor output) {
            // CPU 实现逻辑：OpenMP 并行填充
            size_t total_elements = output->size();
            float* data = static_cast<float*>(output->data());
            #pragma omp parallel for
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = 1.0f;
            }
        },
        true  // 覆盖之前的实现
    );

    // 为所有设备注册通用实现（使用 InfiniOP）
    Ones::dispatcher().registerAll(
        [](Tensor output) {
            // InfiniOP 统一接口实现
            // infiniopOnes(...)
        },
        false
    );
}

// 示例 4: 完整的计算流程
void example_full_workflow() {
    // 初始化输出张量
    Shape shape = {1000, 1000};
    Tensor output = Tensor::empty(shape, DataType::F32, Device::nvidia());

    // 执行填充操作（当前为空，需注册 kernel 后可用）
    // Ones::execute(output);  // 或 ones_(output);

    // 使用填充后的张量进行后续计算
    // ...
}
```

## 5. 实现细节

### 内存管理
- **分配策略**: 使用 `Tensor::empty()` 分配内存，支持主机内存和设备显存
- **内存布局**: 支持连续和步长（strided）张量，由 `Tensor` 类管理元数据
- **对齐要求**: 遵循设备默认对齐（通常为 256 字节用于 SIMD/GPU 优化）

### 并发与线程安全
- **分发器访问**: 静态局部变量使用 C++11 保证的线程安全初始化（Magic Statics）
- **Kernel 注册**: `OpDispatcher::registerDevice()` 内部未显式加锁，多线程同时注册存在数据竞争
  - **建议**: 注册操作应在程序启动阶段单线程完成
- **执行阶段**: `execute()` 方法预期会调用 `context::setDevice()` 设置当前设备，然后调用设备 kernel
  - GPU kernel 通常通过流（stream）管理异步执行
  - CPU kernel 可使用 OpenMP 或 TBB 实现并行填充

### 性能优化
- **算法选择**:
  - CPU: 使用 OpenMP 并行 for 循环，缓存友好的连续内存访问
  - GPU: 使用 CUDA 核函数，利用共享内存和合并访问优化带宽
  - 其他硬件: 通过 InfiniOP 统一接口分发，调用厂商优化库
- **时间复杂度**: O(n)，n 为张量元素总数（必须遍历所有元素）
- **空间复杂度**: O(1) 额外空间（就地修改，无临时缓冲区）
- **带宽利用率**: 纯写操作，理论峰值接近设备内存写带宽（如 GPU: ~900 GB/s）

### 错误处理
- **参数验证**: 当前实现未包含设备一致性检查（对比 Add/Mul 算子）
  - **建议**: 添加 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏检查
- **设备设置**: 未调用 `context::setDevice()`，可能导致 kernel 在错误设备上执行
- **空实现风险**: 当前 `execute()` 函数体为空，调用时产生静默失败
  - **状态**: 代码骨架完整，但缺少实际计算逻辑和 kernel 注册

### 依赖关系
- **内部依赖**:
  - `common/op.hpp`: 提供 `OpDispatcher` 模板类定义
  - `tensor.hpp`: 定义 `Tensor` 类及其内存管理接口
  - `device.hpp`: 提供 `Device::Type` 枚举（CPU、NVIDIA、ASCEND 等）
  - `context/context.hpp`: 提供 `context::setDevice()` 设备切换接口
- **外部依赖**:
  - InfiniOP 库: 提供跨平台的底层算子实现（`infiniopOnes` 等）
  - 运行时库: infinirt 用于设备内存管理和流控制
- **设计模式**:
  - **策略模式（Strategy Pattern）**: 通过 `OpDispatcher` 动态选择不同设备的算法实现
  - **单例模式（Singleton Pattern）**: 分发器使用 Meyer's Singleto n确保全局唯一
  - **工厂模式（Factory Pattern）**: `ones()` 函数作为工厂方法创建新张量
  - **模板方法模式（Template Method）**: `execute()` 定义算法骨架，kernel 提供具体实现

### 扩展性设计
- **设备扩展**: 新增硬件后端只需调用 `registerDevice()` 注册对应 kernel
- **类型支持**: 当前 schema 使用 `Tensor` 作为通用类型，通过 `DataType` 枚举支持多种数值类型
  - 包括: F16、BF16、F32、F64、I8、I32、U8、U32 等
- **Kernel 覆盖**: `registerDevice()` 的 `override_existing` 参数支持运行时替换实现
  - 用途: 性能调优、A/B 测试、回退到兼容实现

### 当前状态与缺失
- **已实现**:
  - ✅ 头文件接口定义完整
  - ✅ 分发器单例模式实现
  - ✅ 符合 InfiniCore 算子标准架构
- **未实现**:
  - ❌ `execute()` 函数体为空
  - ❌ 缺少设备一致性检查
  - ❌ 缺少 kernel 注册代码（参考 `src/infinicore/ops/add/add.cc`）
  - ❌ `ones()` 函数缺少形状和类型参数（对比 `gemm`、`linear` 等算子）
  - ❌ 无 Python 绑定（pybind11）
  - ❌ 无单元测试（参考 `test/infinicore/ops/add.py`）

### 对比参考实现
与 `Add` 算子相比（`src/infinicore/ops/add/add.cc`）：
```cpp
// Add 完整实现模式
void Add::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);  // Ones 缺失
    infinicore::context::setDevice(c->device());      // Ones 缺失
    dispatcher().lookup(c->device().getType())(c, a, b);  // Ones 缺失调用
}

Tensor add(Tensor a, Tensor b) {  // Ones 接口不一致
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_(c, a, b);
    return c;
}
```

**建议的完整实现**（伪代码）：
```cpp
void Ones::execute(Tensor output) {
    infinicore::context::setDevice(output->device());  // 设置设备
    dispatcher().lookup(output->device().getType())(output);  // 调用 kernel
}

Tensor ones(const Shape &shape, DataType dtype, Device device) {
    auto output = Tensor::empty(shape, dtype, device);
    ones_(output);
    return output;
}

void ones_(Tensor output) {
    Ones::execute(output);
}
```

### Kernel 注册示例（参考 Add 实现）
```cpp
namespace infinicore::op::ones_impl {

// CPU 实现
void calculate_cpu(Tensor output) {
    size_t size = output->size();
    float* data = static_cast<float*>(output->data());
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1.0f;
    }
}

// InfiniOP 统一实现
void calculate_infiniop(Tensor output) {
    infiniopOnesDescriptor_t desc;
    infiniopCreateOnesDescriptor(handle, &desc, output->desc());
    // ...
    infiniopOnes(desc, nullptr, 0, output->data(), nullptr, nullptr);
}

// 加载时自动注册
static bool registered = []() {
    Ones::dispatcher().registerDevice(Device::Type::CPU, &calculate_cpu, false);
    Ones::dispatcher().registerAll(&calculate_infiniop, false);
    return true;
}();

} // namespace infinicore::op::ones_impl
```

---

**注**: 该文档基于当前代码状态分析生成，`Ones` 算子处于开发初期，核心执行逻辑和 kernel 注册尚未实现。建议参考 `Add`、`Mul`、`Gemm` 等已完成算子的实现模式补全功能。
