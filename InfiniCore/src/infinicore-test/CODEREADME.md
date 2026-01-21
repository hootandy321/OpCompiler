# InfiniCore Test Suite Core Implementation Documentation

InfiniCore Test Suite 是一个全面的测试框架,用于验证 InfiniCore 深度学习框架的核心功能,包括内存管理、张量操作、神经网络模块、并发安全、性能基准和压力测试。该测试套件采用模块化设计,支持多设备后端(NVIDIA、Cambricon、Ascend、Metax 等),并提供灵活的命令行接口用于执行不同类型的测试。

## 1. Module Structure

- **`main.cc`**: 测试套件主入口,实现命令行参数解析、设备初始化、测试调度和结果汇总
- **`test_runner.h`**: 测试框架基础设施,定义 TestFramework 基类、TestResult 结构、InfiniCoreTestRunner 调度器和张量比较工具
- **`memory_test.h`**: 内存管理测试套件声明,包含基础内存测试、并发测试、异常安全测试、内存泄漏检测、性能测试和压力测试
- **`memory_test.cc`**: 内存管理测试实现,通过 `context::allocateMemory` API 验证分配器正确性、线程安全性和性能指标
- **`test_nn_module.h`**: 神经网络模块测试声明,涵盖 Module 基类、Linear、Embedding、RMSNorm、RoPE 及张量并行功能
- **`test_nn_module.cc`**: 神经网络模块测试实现,验证模块层次结构、参数管理、状态字典加载、前向传播正确性和 TinyLlama 模型构造
- **`test_tensor_destructor.h`**: 张量析构测试声明
- **`test_tensor_destructor.cc`**: 张量析构测试实现,验证张量生命周期管理、内存泄漏检测和不同数据类型/形状的析构行为

## 2. Core Classes

### `ParsedArgs`
- **Location**: `main.cc`
- **Primary Function**: 封装命令行参数解析结果,控制测试执行配置
- **Key Members**:
  - `device_type`: `infiniDevice_t` 类型,指定测试目标设备(CPU/NVIDIA/Cambricon 等)
  - `run_basic/run_concurrency/run_exception_safety/run_memory_leak/run_performance/run_stress/run_module`: 布尔标志,控制测试套件启用
  - `num_threads`: `int` 类型,并发测试线程数(默认 4)
  - `iterations`: `int` 类型,压力测试迭代次数(默认 1000)
- **Core Methods**:
  - `parseArgs(int argc, char* argv[])`: 静态工厂函数,使用迭代解析算法处理命令行参数,支持 `--<device>`, `--test <name>`, `--threads <num>`, `--iterations <num>` 格式,错误时调用 `exit(EXIT_FAILURE)`
- **Lifecycle**: 栈分配值对象,在 `main` 函数开头构造,参数验证失败时直接终止进程

### `InfiniCoreTestRunner`
- **Location**: `test_runner.h`
- **Primary Function**: 测试套件调度器,管理测试生命周期并聚合测试结果
- **Key Members**:
  - `tests_`: `std::vector<std::unique_ptr<TestFramework>>` 类型,存储所有注册的测试实例
- **Core Methods**:
  - `addTest(std::unique_ptr<TestFramework> test)`: 使用移动语义将测试对象添加到测试队列
  - `runAllTests()`: 顺序执行所有测试,调用每个测试的 `run()` 方法,收集 TestResult,调用 `printSummary` 输出通过/失败统计和失败测试详情
- **Lifecycle**: 栈分配对象,在 `main` 函数中构造,测试执行完毕后自动析构

### `TestFramework`
- **Location**: `test_runner.h`
- **Primary Function**: 所有测试类的抽象基类,定义测试接口和通用工具方法
- **Key Members**:
  - 无成员变量,纯接口类
- **Core Methods**:
  - `virtual TestResult run() = 0`: 纯虚函数,子类实现具体测试逻辑
  - `virtual std::string getName() const = 0`: 纯虚函数,返回测试名称用于日志输出
  - `measureTime(const std::string& test_name, Func&& func)`: 模板方法,使用 `std::chrono::high_resolution_clock` 测量函数执行时间,捕获异常并返回包含时长的 TestResult
  - `logTestStart/logTestResult`: 辅助方法,输出标准格式的测试日志
- **Lifecycle**: 多态基类,子类通过 `std::make_unique` 构造并传递给 `InfiniCoreTestRunner::addTest`

### `BasicMemoryTest`
- **Location**: `memory_test.h/cc`
- **Primary Function**: 验证基础内存分配 API 的正确性,包括常规分配、固定内存分配和内存访问测试
- **Key Members**:
  - 继承自 `TestFramework`,无额外成员
- **Core Methods**:
  - `TestResult run() override`: 调用 `measureTime` 执行测试逻辑,验证 `context::allocateMemory(1024)` 返回非空指针,检查 `memory->size() == 1024`,对 CPU 设备执行 `std::memset` 写入和读取验证,测试 `context::allocatePinnedHostMemory(512)` 并验证 `is_pinned()` 属性
- **Implementation Details**:
  - 使用 `spdlog::debug` 输出详细日志,对于 GPU 设备跳过直接内存访问测试,检查 `current_device.getType() != Device::Type::CPU` 条件
  - 使用 `std::memset(data, 0xAB, 1024)` 写入模式字节,循环验证每个字节的值

### `ConcurrencyTest`
- **Location**: `memory_test.h/cc`
- **Primary Function**: 验证内存分配器在多线程环境下的线程安全性和数据竞争处理
- **Key Members**:
  - 无额外成员
- **Core Methods**:
  - `TestResult run() override`: 顺序执行三个子测试,目前仅启用 `testConcurrentAllocations`
  - `testConcurrentAllocations()`: 使用 `std::vector<std::thread>` 启动 8 个线程,每个线程执行 100 次随机大小分配(64-1088 字节),使用 `std::atomic<int> success_count` 和 `failure_count` 统计结果,验证 `success_count.load() == 8 * 100`
  - `testConcurrentDeviceSwitching()`: (已注释) 在多设备环境下测试 `context::setDevice` 的线程安全性
  - `testMemoryAllocationRace()`: (已注释) 高频率分配/释放竞争测试,16 线程 * 1000 次分配,每 10 次分配释放一次
- **Concurrency**:
  - 使用 `std::thread` 原生线程,通过 `thread.join()` 等待所有线程完成
  - 使用 `std::atomic<int>` 无锁计数器,避免 `std::mutex` 开销
  - 使用 `std::this_thread::sleep_for(std::chrono::microseconds(1))` 增加竞争概率

### `MemoryLeakTest`
- **Location**: `memory_test.h/cc`
- **Primary Function**: 检测内存泄漏,通过 `MemoryLeakDetector` 单例跟踪分配/释放操作
- **Key Members**:
  - 无直接成员,依赖全局单例 `MemoryLeakDetector::instance()`
- **Core Methods**:
  - `testBasicLeakDetection()`: 调用 `MemoryLeakDetector::instance().reset()` 清零,分配 100 个 1024 字节内存块,调用 `memories.clear()` 触发析构,等待 100ms 后检查 `getLeakedMemory() == 0`
  - `testCrossDeviceLeakDetection()`: 在设备 A 分配固定内存,切换到设备 B 后释放,验证跨设备释放无泄漏
  - `testExceptionLeakDetection()`: 在 try-catch 块中分配内存后抛出异常,验证异常路径下的内存释放
- **Memory Management**:
  - 使用 `std::shared_ptr<Memory>` 自动管理生命周期,RAII 模式确保异常安全

### `MockLinearModule`
- **Location**: `test_nn_module.h`
- **Primary Function**: 模拟 PyTorch `nn.Linear` 的测试模块,用于验证参数注册和状态字典机制
- **Key Members**:
  - `INFINICORE_NN_PARAMETER(weight)`: 宏声明的参数对象,形状 `[output_size, input_size]`
  - `INFINICORE_NN_PARAMETER(bias)`: 宏声明的偏置参数,形状 `[output_size]`
  - `input_size_/output_size_`: `int` 类型,存储模块维度
  - `tp_dim_/tp_rank_/tp_size_`: `Size` 类型(实际上为 `size_t`),支持张量并行分片
- **Core Methods**:
  - `MockLinearModule(int input_size, int output_size, const Device& device, Size tp_dim, Size tp_rank, Size tp_size)`: 构造函数,使用 `INFINICORE_NN_PARAMETER_INIT` 宏初始化参数,传递张量并行参数
  - `Tensor forward(const Tensor& input)`: 占位前向传播,当前直接返回输入(实际应执行矩阵乘法)
  - `Tensor get_weight()/get_bias()`: 通过 `state_dict()` 查找参数并返回,失败时抛出 `std::runtime_error`
- **Lifecycle**: 栈分配或 `std::make_unique` 动态分配,参数通过宏自动注册到 `state_dict()`

### `NNModuleTest`
- **Location**: `test_nn_module.h/cc`
- **Primary Function**: 神经网络模块综合测试套件,覆盖 12 个测试场景,验证 Module 机制、参数管理、前向传播和模型构造
- **Key Members**:
  - 继承自 `TestFramework`,无额外成员
- **Core Methods**:
  - `testBasicModuleCreation()`: 测试 1a/1b/1c 合并,验证 `MockLinearModule` 构造、参数注册(2 个参数)、形状验证(weight `[4,8]`, bias `[4]`)、`load_parameter_` 和 `load_state_dict` 功能
  - `testTensorParallelParameters()`: 测试张量并行参数分片,创建 `tp_size=4, tp_dim=0/1` 的参数,验证分片后形状(dim0: `[4,4]` vs `[8,2]`),使用 `load_parameter_from_blob` 从完整权重加载数据,通过 `narrow` 切片验证分片正确性
  - `testParalleLinear()`: 测试 `ColumnParallelLinear` 和 `RowParallelLinear`,验证 tp_dim=0 时 weight/bias 均分片(`[8,64]`, `[8]`),tp_dim=1 时仅 weight 分片(`[32,16]`, `[32]`)
  - `testLoadStateDict()`: 测试 2 层嵌套模块(DeepParentModule -> DeepChildModule -> DeepGrandchildModule),验证参数命名规则(`layer1.sublayer.sublayer.weight`),加载全 1.0 权重后使用 `tensorsAllClose` 比对数值
  - `testModuleHierarchy()`: 测试 3 层嵌套(RootModule -> Layer1Module -> Layer2Module -> MockLinearModule),验证参数数量(6 个)和命名,测试 `INFINICORE_NN_MODULE_VEC` 宏注册的向量模块(layers.0/1/2)
  - `testParameterLoading()`: 测试从 blob 加载参数,使用 `load_parameter_from_blob` 将 CPU 内存数据加载到参数张量
  - `testModuleLinear()`: 全面的 Linear 模块测试,分别测试有/无 bias 情况,验证 `in_features/out_features/has_bias` 属性,执行前向传播并验证输出形状(`[2,4]`),使用 Naive 实现逐步骤计算(matmul + bias broadcast + residual add)并与 InfiniCore 结果进行 `tensorsAllClose` 比较
  - `testModuleEmbedding()`: 测试 Embedding 模块,验证 `num_embeddings/embedding_dim/padding_idx` 属性,测试单索引、批索引、2D 索引(`[2,4,64]`)前向传播,验证相同索引返回相同嵌入
  - `testModuleRMSNorm()`: 测试 RMSNorm 模块,验证 `normalized_shape/eps` 属性,测试 2D/3D 输入归一化,验证归一化后形状不变,测试不同 hidden_size(128/4096)
  - `testModuleRoPE()`: 测试 RoPE 模块,验证 `head_dim/max_seq_len/theta/algo` 属性,测试 3D 输入(`[32,32,128]`)旋转位置编码,支持 GPT_J/GPT_NEOX 两种算法,验证奇数 head_dim 抛出 `std::invalid_argument`,测试不同 head_dim(64/128/256)
  - `testDtypeAssertion()`: 测试 dtype 断言,验证加载匹配/不匹配 dtype 的参数,预期不匹配时抛出包含 "dtype mismatch" 的 `std::runtime_error`
  - `testTinyLlamaConstruction()`: 综合测试,构造完整的 TinyLlama-1.1B 模型(22 层,32000 vocab,2048 hidden,GQA 4/32 头),验证参数数量(200 个)、形状匹配、权重加载成功
- **Implementation Details**:
  - 使用 `spdlog` 系列宏(info/debug/error)输出详细测试日志
  - 使用 `tensorsAllClose` 进行浮点数比较,容差 `rtol=1e-5, atol=1e-5`
  - 使用 `Tensor::from_blob` 从 CPU 内存创建张量,使用 `->to(device)` 转移设备
  - 使用 `->narrow({{dim, start, length}})` 切片操作
  - 使用 `INFINICORE_NN_MODULE` 宏声明子模块,`INFINICORE_NN_MODULE_INIT` 宏初始化子模块

### `TensorDestructorTest`
- **Location**: `test_tensor_destructor.h/cc`
- **Primary Function**: 验证张量析构函数正确调用和内存释放
- **Key Members**:
  - 继承自 `TestFramework`,无额外成员
- **Core Methods**:
  - `testBasicTensorDestruction()`: 在作用域内创建 `Tensor::empty({2,3}, F32, CPU)`,验证形状和数据类型,离开作用域后验证析构函数调用(通过日志输出)
  - `testMultipleTensorDestruction()`: 创建 4 个不同形状和类型的张量(F32/F64/I32/F16),验证全部创建成功,向量析构时触发批量析构
  - `testDifferentDataTypes()`: 测试 8 种数据类型(F32/F64/F16/I32/I64/I8/U8/BOOL),逐个创建并析构,验证 dtype 匹配
  - `testDifferentShapes()`: 测试 8 种形状(1D 到 5D,大尺寸张量),包括 `[1000]`, `[100,100]`, `[10,10,10,10]`
  - `testTensorFromBlob()`: 使用 `Tensor::from_blob` 从已有数据创建张量,验证形状和 dtype,验证析构不损坏原始数据
  - `testStridedTensor()`: 创建 `Tensor::empty({4,4})`,使用 `->narrow({{0,0,2},{1,0,2}})` 创建切片视图,验证视图形状 `[2,2]`
  - `testMemoryLeakDetection()`: 使用 `MemoryLeakDetector::instance().reset()` 重置,循环创建 100 个张量并析构,验证 `final_leaks <= initial_leaks`
  - `testTensorCopyDestruction()`: 创建张量后使用赋值运算符复制,验证两个共享同一底层对象,离开作用域后正确释放

### `MemoryLeakDetector`
- **Location**: `memory_test.h`
- **Primary Function**: 单例内存泄漏检测器,跟踪所有内存分配和释放操作
- **Key Members**:
  - `allocations_`: `std::unordered_map<void*, size_t>` 类型,记录指针到大小的映射
  - `total_allocated_`: `size_t` 类型,累计分配字节数
  - `mutex_`: `mutable std::mutex` 类型,保护内部数据的线程安全
- **Core Methods**:
  - `static MemoryLeakDetector& instance()`: Meyer's Singleton 模式,使用静态局部变量确保线程安全初始化
  - `recordAllocation(void* ptr, size_t size)`: 加锁后插入 `allocations_[ptr] = size`,累加 `total_allocated_`
  - `recordDeallocation(void* ptr)`: 加锁后查找 `allocations_`,若存在则减去大小并擦除条目
  - `getLeakedMemory() const`: 加锁后返回 `total_allocated_`
  - `getLeakCount() const`: 加锁后返回 `allocations_.size()`
  - `reset()`: 清空 `allocations_` 并重置 `total_allocated_` 为 0
- **Concurrency**: 使用 `std::lock_guard<std::mutex>` 在所有公共方法中确保线程安全

## 3. API Interface

```cpp
// Test Framework Core APIs
class TestFramework {
public:
    virtual TestResult run() = 0;
    virtual std::string getName() const = 0;

protected:
    template <typename Func>
    TestResult measureTime(const std::string& test_name, Func&& func);
};

// Test Runner API
class InfiniCoreTestRunner {
public:
    void addTest(std::unique_ptr<TestFramework> test);
    std::vector<TestResult> runAllTests();
};

// Tensor Comparison Utility (for testing numerical correctness)
inline bool tensorsAllClose(const infinicore::Tensor &actual,
                            const infinicore::Tensor &expected,
                            double rtol = 1e-5,
                            double atol = 1e-5);
// Compares two tensors elementwise with tolerance, automatically moves to CPU,
// reports up to 10 mismatches with coordinates, supports F32 dtype only

// Memory Leak Detection API
class MemoryLeakDetector {
public:
    static MemoryLeakDetector& instance();
    void recordAllocation(void *ptr, size_t size);
    void recordDeallocation(void *ptr);
    size_t getLeakedMemory() const;
    size_t getLeakCount() const;
    void reset();
};

// Command Line Parsing (main.cc)
ParsedArgs parseArgs(int argc, char *argv[]);
// Supports: --<device>, --test <name>, --threads <num>, --iterations <num>, --help
```

## 4. Usage Example

```cpp
// Example: Running the InfiniCore Test Suite with custom configuration

// 1. Include the test infrastructure
#include "test_runner.h"
#include "memory_test.h"
#include "test_nn_module.h"
#include "test_tensor_destructor.h"
#include <infinicore.hpp>

// 2. Initialize InfiniCore context
infinicore::context::setDevice(
    infinicore::Device(infinicore::Device::Type::CPU, 0)
);

// 3. Create test runner
infinicore::test::InfiniCoreTestRunner runner;

// 4. Add individual test suites
runner.addTest(std::make_unique<infinicore::test::BasicMemoryTest>());
runner.addTest(std::make_unique<infinicore::test::TensorDestructorTest>());
runner.addTest(std::make_unique<infinicore::test::NNModuleTest>());
runner.addTest(std::make_unique<infinicore::test::ConcurrencyTest>());

// 5. Run all tests and collect results
auto results = runner.runAllTests();

// 6. Analyze results
size_t passed = 0, failed = 0;
for (const auto &result : results) {
    if (result.passed) {
        passed++;
    } else {
        failed++;
        std::cout << "Failed: " << result.test_name
                  << " - " << result.error_message << std::endl;
    }
}

std::cout << "Summary: " << passed << " passed, " << failed << " failed" << std::endl;

// Example: Using tensor comparison in custom tests
infinicore::Tensor output = module.forward(input);
infinicore::Tensor expected = infinicore::Tensor::ones({2, 4}, ...);
if (!infinicore::test::tensorsAllClose(output, expected, 1e-5, 1e-5)) {
    std::cerr << "Output does not match expected values" << std::endl;
}

// Example: Detecting memory leaks in custom allocations
infinicore::test::MemoryLeakDetector::instance().reset();
{
    auto tensor = infinicore::Tensor::empty({1000, 1000}, ...);
    // Tensor automatically deallocated at end of scope
}
size_t leaked = infinicore::test::MemoryLeakDetector::instance().getLeakedMemory();
if (leaked > 0) {
    std::cerr << "Memory leak detected: " << leaked << " bytes" << std::endl;
}
```

## 5. Implementation Details

**Memory Management**:
- 使用 RAII 模式,所有内存对象通过 `std::shared_ptr<Memory>` 管理,确保异常安全
- 测试中使用作用域 `{}` 验证析构函数调用时机
- `MemoryLeakDetector` 单例跟踪所有分配/释放,通过 `unordered_map` 记录指针到大小的映射
- 对于 GPU 设备,测试跳过直接 CPU 内存访问,检查 `device.getType() != Device::Type::CPU` 条件

**Concurrency**:
- 使用 `std::thread` 原生线程 API,避免第三方线程库依赖
- 使用 `std::atomic<int>` 无锁计数器统计成功/失败次数,避免 `std::mutex` 开销
- `MemoryLeakDetector` 内部使用 `std::lock_guard<std::mutex>` 保护 `unordered_map` 访问
- 并发测试中使用 `std::this_thread::sleep_for` 增加竞争条件概率
- 线程函数通过 lambda 捕获 `&` 引用外部变量,确保共享状态更新

**Performance**:
- 使用 `std::chrono::high_resolution_clock` 测量微秒级执行时间
- 性能测试设定阈值:单线程分配 < 100μs,并发分配 < 200μs,内存拷贝带宽 > 100MB/s
- 压力测试执行 100,000 次高频分配,验证分配器稳定性
- 大内存分配测试分配 10 个 100MB 块,验证内存管理能力
- 使用 `reserve()` 预分配 `std::vector` 容量,减少动态扩容开销

**Error Handling**:
- 所有测试函数返回 `TestResult` 结构,包含 `passed` 布尔值和 `error_message` 字符串
- 使用 `measureTime` 模板函数捕获 `std::exception`,自动转换为失败结果
- 异常测试验证预期异常类型(`std::runtime_error`, `std::invalid_argument`)和错误消息内容
- 使用 `spdlog::error` 输出详细错误信息,包含参数名称、期望值和实际值
- 异常安全测试验证 `try-catch` 块中资源正确释放

**Dependencies**:
- **InfiniCore Core**: 依赖 `infinicore.hpp`, `infinicore/tensor.hpp`, `infinicore/context/context.hpp`
- **InfiniCore NN**: 依赖 `infinicore/nn/module.hpp`, `infinicore/nn/linear.hpp`, `infinicore/nn/embedding.hpp`, `infinicore/nn/rmsnorm.hpp`, `infinicore/nn/rope.hpp`
- **InfiniCore Ops**: 依赖 `infinicore/ops.hpp` 用于张量操作(matmul, add_)
- **spdlog**: 使用 `spdlog/spdlog.h` 进行结构化日志输出(spdlog::info/debug/error/warn)
- **Standard Library**: 使用 `<chrono>`, `<thread>`, `<atomic>`, `<mutex>`, `<vector>`, `<memory>`, `<unordered_map>`, `<fstream>`, `<algorithm>`

**Design Patterns**:
- **Singleton Pattern**: `MemoryLeakDetector` 使用 Meyer's Singleton(静态局部变量),确保线程安全初始化
- **Strategy Pattern**: `TestFramework` 定义测试接口,子类实现不同测试策略(BasicMemory/Concurrency/NNModule 等)
- **Template Method Pattern**: `measureTime` 定义测试执行模板,子测试通过 lambda 传递具体逻辑
- **Factory Pattern**: `main.cc` 中使用 `std::make_unique<ConcreteTest>()` 工厂方法创建测试对象
- **Composite Pattern**: NN Module 测试中使用嵌套 `INFINICORE_NN_MODULE` 宏构建层次化模块(TinyLlama: Model -> Layers -> SelfAttn/MLP)
- **RAII Pattern**: 所有资源(内存、张量、模块)通过智能指针自动管理,确保异常安全
- **Macro Metaprogramming**: 使用 `INFINICORE_NN_PARAMETER`, `INFINICORE_NN_MODULE`, `INFINICORE_NN_MODULE_VEC` 等宏简化重复代码,自动生成参数注册逻辑

**Algorithm Details**:
- **张量比较算法** (`tensorsAllClose`): 预计算步长(stride)用于线性索引到多维坐标转换,逐元素计算 `fabs(a-b)`,与 `atol + rtol * fabs(b)` 比较,记录前 10 个不匹配点的坐标和最大差异位置
- **张量并行参数分片**: 根据 `tp_dim` 和 `tp_rank` 计算分片形状,`dim0`: `shape[0] / tp_size`, `dim1`: `shape[1] / tp_size`,使用 `narrow` 从完整权重提取对应分片
- **命令行参数解析**: 迭代 `argv[i]`,使用 `if-else if` 链匹配参数名,对于需要参数值的选项(`--test`, `--threads`)使用 `++i` 获取下一个参数,错误时调用 `exit(EXIT_FAILURE)`
- **并发测试统计**: 使用 `std::atomic<int>` 计数器,每个线程执行 `success_count++` 或 `failure_count++`,主线程 `join()` 后调用 `load()` 获取最终值
- **内存泄漏检测**: 在分配时调用 `recordAllocation(ptr, size)`,析构时调用 `recordDeallocation(ptr)`,`reset()` 后所有泄漏为新增泄漏
