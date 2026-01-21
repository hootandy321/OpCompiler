# Add Operation Core Implementation Documentation

此模块实现了 Infini 框架中的张量加法运算,提供跨多种计算后端(CUDA、CPU、Kunlun、Ascend 等)的统一接口,采用策略模式实现硬件无关的算子分发,并集成 InfiniOP 库进行高性能计算。

## 1. Module Structure

- **`add.cc`**: 核心算子实现,定义 Add 类的执行逻辑与分发器管理
- **`add_infiniop.cc`**: InfiniOP 后端实现,提供基于 libinfiniop 的具体计算实现,包含算子描述符缓存机制
- **`include/infinicore/ops/add.hpp`**: 公共 API 接口定义,声明 Add 类及相关工具函数

## 2. Core Classes

### `Add`
- **Location**: `include/infinicore/ops/add.hpp`, `add.cc`
- **Primary Function**: 提供张量加法运算的静态接口,管理设备特定的实现分发器
- **Key Members**:
  - `schema`: 类型别名 `void (*)(Tensor, Tensor, Tensor)`,定义算子函数签名
  - `dispatcher_`: 静态局部变量 `common::OpDispatcher<Add::schema>`,按设备类型存储实现函数指针
- **Core Methods**:
  - `execute(Tensor c, Tensor a, Tensor b)`: 执行加法运算 c = a + b,首先验证所有张量位于同一设备,设置目标设备上下文,然后通过分发器查找并调用设备特定的实现函数
  - `dispatcher()`: 返回单例分发器引用,使用 Meyer's Singleton 确保线程安全的延迟初始化
- **Lifecycle**: 采用静态局部变量实现 Singleton 模式,首次调用 `dispatcher()` 时构造,程序结束时自动销毁

### `OpCache<size_t, infiniopAddDescriptor_t>`
- **Location**: `add_infiniop.cc` (thread_local 实例)
- **Primary Function**: 为 InfiniOP 算子描述符提供线程局部缓存,避免重复创建描述符的开销
- **Key Members**:
  - `caches`: `thread_local OpCache` 实例,容量为 100,自定义析构函数负责调用 `infiniopDestroyAddDescriptor`
  - 析构 lambda: 封装 `infiniopDestroyAddDescriptor` 调用,确保描述符正确释放
- **Core Methods**:
  - `getCache(Device)`: 获取指定设备的 LRU 缓存实例,内部按 `(Device::Type, device_index)` 二维索引组织缓存
  - `get(seed)`: 从缓存查找算子描述符,返回 `std::optional<infiniopAddDescriptor_t>`
  - `put(seed, desc)`: 将新创建的描述符存入缓存
- **Lifecycle**: 线程局部存储,每个线程拥有独立缓存,线程退出时自动清理

### `calculate` (InfiniOP 实现)
- **Location**: `add_infiniop.cc`
- **Primary Function**: InfiniOP 后端的实际计算函数,处理描述符创建/缓存、工作空间分配、内核调度
- **Algorithm Flow**:
  1. **哈希计算**: 使用 `hash_combine(c, b, a)` 生成操作特征签名,包含数据类型、形状、步长信息
  2. **缓存查询**: 从 thread_local 缓存查找描述符,未命中时创建新描述符并缓存
  3. **描述符创建**: 调用 `infiniopCreateAddDescriptor` 生成操作描述符,需要 InfiniOP Handle 和张量描述符
  4. **工作空间分配**: 通过 `infiniopGetAddWorkspaceSize` 查询所需临时内存大小,分配设备内存
  5. **内核启动**: 调用 `infiniopAdd` 执行实际计算,传入工作空间指针、数据指针、CUDA 流
- **Complexity**: 描述符创建为 O(1) 哈希查找,内核执行复杂度由底层 InfiniOP 实现决定(通常为 O(n) 其中 n 为元素数量)

## 3. API Interface

```cpp
// 高级 API: 自动分配输出张量
Tensor add(Tensor a, Tensor b);
// 功能: 执行 c = a + b,自动创建与 a 相同形状、类型、设备的输出张量 c
// 参数: a - 第一个操作数张量, b - 第二个操作数张量
// 返回: 包含运算结果的新张量
// 约束: a 和 b 必须位于同一设备,形状需可广播

// 低级 API: 预分配输出张量
void add_(Tensor c, Tensor a, Tensor b);
// 功能: 执行 c = a + b,写入预分配的输出张量 c
// 参数: c - 输出张量(必须预先分配), a, b - 输入张量
// 约束: c 的形状和类型必须与运算结果匹配,所有张量需同设备

// 运算符重载
Tensor operator+(Tensor a, Tensor b);
// 功能: 提供 a + b 语法糖,内部调用 add(a, b)

// 内部调度接口(Add 类)
class Add {
    static void execute(Tensor c, Tensor a, Tensor b);
    // 功能: 内部执行入口,通过 OpDispatcher 分发到设备特定实现

    static common::OpDispatcher<schema> &dispatcher();
    // 功能: 获取单例分发器引用,用于注册和查找设备实现
};
```

## 4. Usage Example

```cpp
#include "infinicore/ops/add.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 初始化上下文(例如 CUDA)
context::initialize(Device::Type::CUDA);
Device device(0);

// 创建输入张量
auto a = Tensor::arange({2, 3}, DataType::FLOAT32, device);  // [[0,1,2], [3,4,5]]
auto b = Tensor::ones({2, 3}, DataType::FLOAT32, device);    // [[1,1,1], [1,1,1]]

// 方式 1: 高级 API(自动分配输出)
Tensor c = op::add(a, b);  // c = [[1,2,3], [4,5,6]]

// 方式 2: 运算符重载
Tensor d = a + b;          // 等价于 add(a, b)

// 方式 3: 低级 API(预分配输出,用于内存优化场景)
Tensor e = Tensor::empty({2, 3}, DataType::FLOAT32, device);
op::add_(e, a, b);         // e = [[1,2,3], [4,5,6]]

// 方式 4: 直接调用 Add 类(高级用法)
op::Add::execute(e, a, b); // 等价于 add_(e, a, b)

// 设备间操作会触发运行时断言
auto cpu_tensor = Tensor::ones({2, 3}, DataType::FLOAT32, Device(Device::Type::CPU, 0));
// Tensor result = op::add(a, cpu_tensor);  // 错误! INFINICORE_ASSERT_TENSORS_SAME_DEVICE 失败
```

## 5. Implementation Details

### Memory Management
- **输出张量分配**: `add()` 使用 `Tensor::empty()` 在目标设备上分配连续内存,形状由输入张量决定
- **工作空间管理**: InfiniOP 实现通过 `infiniopGetAddWorkspaceSize` 动态查询内核所需的临时内存大小,使用 `context::allocateMemory` 分配设备内存,支持异步计算流
- **描述符生命周期**: 使用 RAII 包装的 OpCache 管理描述符,缓存满时自动调用析构函数释放旧描述符,避免内存泄漏

### Concurrency
- **线程局部缓存**: 使用 `thread_local OpCache` 确保每个线程拥有独立的描述符缓存,避免多线程竞争
- **设备上下文隔离**: OpCache 按 `(Device::Type, device_index)` 二维索引组织,不同设备的缓存互不干扰
- **CUDA 流支持**: `infiniopAdd` 调用通过 `context::getStream()` 获取当前流的句柄,支持同一设备上的多流并发执行
- **无锁设计**: 读操作(缓存查找)通过 std::optional 实现,写操作(缓存插入)仅在缓存未命中时发生,线程安全性由线程局部存储保证

### Performance
- **描述符缓存**: 通过哈希特征值缓存 InfiniOP 描述符,避免重复调用来昂贵的 `infiniopCreateAddDescriptor`(涉及设备内核编译、参数验证等)
- **零拷贝优化**: `add_()` 允许用户预分配输出张量,支持原地计算或输出复用,减少内存分配开销
- **延迟注册**: 使用静态局部变量 + lambda 实现自动注册,程序启动时无需显式初始化
- **LRU 淘汰策略**: OpCache 默认容量为 100,采用 LRU 算法淘汰最久未使用的描述符,平衡内存占用与缓存命中率

### Error Handling
- **运行时断言**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 在执行前验证所有张量位于同一设备,防止跨设备非法访问
- **错误码检查**: 所有 InfiniOP API 调用均通过 `INFINICORE_CHECK_ERROR` 宏包装,将错误码转换为 C++ 异常
- **设备状态管理**: 执行前调用 `context::setDevice()` 确保当前设备上下文正确,避免 CUDA 错误
- **描述符空指针检查**: 缓存查询返回 `std::optional`,明确区分"缓存命中"与"缓存未命中"两种情况

### Dependencies
- **外部依赖**:
  - `libinfiniop`: 提供 InfiniOP 算子库接口(`infiniop.h`),包括描述符管理、工作空间查询、内核调度
  - `CUDA Runtime`: 通过 InfiniOP 间接依赖,用于 GPU 内存管理和内核启动
- **内部模块**:
  - `infinicore/context`: 提供设备管理、内存分配、流管理功能
  - `infinicore/tensor`: 定义张量数据结构及形状/步长/数据类型查询
  - `infinicore/ops/common`: 提供 OpDispatcher(分发器)、OpCache(缓存)、LRUCache(基础缓存)
  - `infinicore/common/hash`: 提供 `hash_combine` 函数用于生成操作特征哈希
- **编译单元依赖**: `add.cc` 依赖 `utils.hpp`(断言宏),`add_infiniop.cc` 依赖多个头文件实现缓存机制

### Design Patterns
- **Strategy Pattern**: OpDispatcher 根据 `Device::Type` 动态选择具体实现策略(CUDA、CPU、Kunlun 等),运行时可替换实现
- **Singleton Pattern**: Add::dispatcher() 使用 Meyer's Singleton,确保全局唯一分发器实例,利用 C++11 保证线程安全
- **Factory Pattern**: `infiniopCreateAddDescriptor` 作为工厂函数,根据张量描述符创建特定配置的算子描述符
- **Cache-Aside Pattern**: `calculate` 函数实现缓存旁路模式,先查缓存,未命中再创建并缓存
- **RAII Pattern**: OpCache 析构函数自动清理所有缓存的描述符,异常安全
- **Template Method**: `add()` 定义高级算法骨架(分配输出->调用低级API),`add_()` 实现具体执行
