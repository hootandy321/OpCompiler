# CUDA Metrics 性能测量模块核心实现文档

本模块是 InfiniPerf 基准测试框架中用于 NVIDIA GPU 性能指标测量的核心组件，通过 NVIDIA PerfWorks (NVPW) 和 CUPTI (CUDA Performance Tools Interface) API 实现对 GPU 硬件计数器的采集、配置和评估。该模块支持自动化的性能指标收集，包括 DRAM 带宽、L2 缓存命中率、计算单元利用率等关键 GPU 性能指标。

## 1. 模块结构

- **`Eval.hpp`**: GPU 指标评估引擎，负责从计数器数据图像中提取和计算指标值
- **`Metric.hpp`**: 指标配置生成器，创建性能测量所需的配置图像和计数器数据前缀
- **`measureMetricPW.hpp`**: 测量会话管理器，实现性能测试的开始/停止控制流程
- **`Parser.h` / `Parser.hpp`**: 指标名称解析器，解析指标字符串中的修饰符（孤立模式、实例保留等）
- **`Utils.h`**: 错误处理工具集，提供 NVPW 状态码到可读字符串的转换
- **`ScopeExit.h`**: RAII 作用域退出辅助工具，确保资源自动释放
- **`measureMetricPW.cpp`**: C++ 接口实现（仅包含头文件包含）
- **`pythonInterface.cpp`**: Python 绑定层，为 Python 提供测量接口

## 2. 核心类与数据结构

### `MetricNameValue` (Eval.hpp, 第14-20行)
- **Location**: `Eval.hpp`
- **Primary Function**: 存储单个指标在多个测量范围内的名称与数值对
- **Key Members**:
  - `metricName` (std::string): 指标名称，如 "dram__bytes_read.sum"
  - `numRanges` (int): 测量范围数量（通常对应多个 kernel 启动或代码段）
  - `rangeNameMetricValueMap` (vector<pair<string, double>>): 每个范围的描述符（如 "kernel1/pass1"）到指标值的映射
- **Lifecycle**: 值对象，在 `GetMetricGpuValue` 中构造并填充

### `NVPW_MetricsEvaluator` (外部类型，通过 NVPW API 操作)
- **Location**: 通过 `NVPW_CUDA_MetricsEvaluator_Initialize` 创建 (Eval.hpp 第78-83行)
- **Primary Function**: NVIDIA 提供的指标评估器，将原始计数器数据转换为语义化指标值
- **Initialization**:
  1. 调用 `NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize` 计算临时缓冲区大小
  2. 分配 `scratchBuffer` 向量
  3. 调用 `NVPW_CUDA_MetricsEvaluator_Initialize` 初始化评估器，传入芯片名称和计数器可用性图像
- **Destruction**: 通过 `NVPW_MetricsEvaluator_Destroy` 释放 (Eval.hpp 第151-153行)
- **Key Operations**:
  - `ConvertMetricNameToMetricEvalRequest`: 将指标名称转换为评估请求
  - `SetDeviceAttributes`: 从计数器数据图像中设置设备属性
  - `EvaluateToGpuValues`: 执行指标计算，输出 GPU 级别的数值

### `NVPW_RawMetricsConfig` (外部类型，通过 NVPW API 操作)
- **Location**: 通过 `NVPW_CUDA_RawMetricsConfig_Create_V2` 创建 (Metric.hpp 第85-90行)
- **Primary Function**: 原始指标配置对象，生成用于配置 CUPTI profiler 的配置图像
- **Initialization**:
  1. 通过 `NVPW_CUDA_RawMetricsConfig_Create_V2` 创建，指定芯片名称
  2. 调用 `SetCounterAvailability` 设置计数器可用性图像（可选）
  3. 调用 `BeginPassGroup` 开始新的采集通道组
  4. 调用 `AddMetrics` 添加原始指标请求列表
  5. 调用 `EndPassGroup` 结束通道组
  6. 调用 `GenerateConfigImage` 生成配置图像
- **Destruction**: 通过 `NVPW_RawMetricsConfig_Destroy` 释放，使用 `SCOPE_EXIT` 确保异常安全 (Metric.hpp 第100-102行)

### `NVPW_CounterDataBuilder` (外部类型，通过 NVPW API 操作)
- **Location**: 通过 `NVPW_CUDA_CounterDataBuilder_Create` 创建 (Metric.hpp 第141-144行)
- **Primary Function**: 计数器数据构建器，生成用于初始化计数器数据图像的前缀数据
- **Initialization**:
  1. 创建 `CounterDataBuilder` 实例
  2. 调用 `AddMetrics` 添加原始指标请求
  3. 调用 `GetCounterDataPrefix` 获取前缀图像（两次调用：第一次获取大小，第二次获取数据）
- **Destruction**: 通过 `NVPW_CounterDataBuilder_Destroy` 释放，使用 `SCOPE_EXIT` 保证清理 (Metric.hpp 第146-148行)

### `ScopeExit<T>` (ScopeExit.h, 第5-11行)
- **Location**: `ScopeExit.h`
- **Primary Function**: RAII (Resource Acquisition Is Initialization) 辅助类，确保在作用域退出时执行指定的清理代码
- **Key Members**:
  - `t` (T): 存储可调用对象（通常是 lambda 表达式）
- **Core Methods**:
  - 构造函数: 接收可调用对象并存储
  - 析构函数: 执行存储的可调用对象 `t()`
- **Usage Pattern**: 配合 `SCOPE_EXIT` 宏使用，例如在 `Metric.hpp` 第102行确保 `RawMetricsConfig` 在函数退出前被销毁

## 3. API 接口

### 指标配置接口 (Metric.hpp)

```cpp
bool GetRawMetricRequests(
    std::string chipName,
    const std::vector<std::string>& metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    const uint8_t* pCounterAvailabilityImage);
// 将高级指标名称（如 "dram__bytes_read.sum"）解析为原始指标请求列表
// 通过 NVPW MetricsEvaluator 查询每个指标的原始依赖项（计数器）
// 输出可用于配置 Profiler 的原始指标请求结构体数组
```

```cpp
bool GetConfigImage(
    std::string chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* pCounterAvailabilityImage);
// 生成 CUPTI Profiler 配置图像
// 算法流程：
// 1. 调用 GetRawMetricRequests 获取原始指标依赖
// 2. 创建 RawMetricsConfig 对象
// 3. 将原始指标请求添加到配置
// 4. 生成配置图像并填充到 configImage 向量
// 5. 使用 SCOPE_EXIT 确保 RawMetricsConfig 被销毁
```

```cpp
bool GetCounterDataPrefixImage(
    std::string chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataImagePrefix,
    const uint8_t* pCounterAvailabilityImage = NULL);
// 生成计数器数据图像的前缀部分
// 前缀包含指标采集所需的元数据和初始化信息
// 算法流程：
// 1. 获取原始指标请求
// 2. 创建 CounterDataBuilder
// 3. 添加指标请求到构建器
// 4. 提取前缀图像（两次调用模式：首次查询大小，二次复制数据）
```

### 指标评估接口 (Eval.hpp)

```cpp
bool GetMetricGpuValue(
    std::string chipName,
    const std::vector<uint8_t>& counterDataImage,
    const std::vector<std::string>& metricNames,
    std::vector<MetricNameValue>& metricNameValueMap,
    const uint8_t* pCounterAvailabilityImage = NULL);
// 从计数器数据图像中提取并计算指标值，结果按范围和指标组织
// 算法流程：
// 1. 验证 counterDataImage 非空
// 2. 初始化 MetricsEvaluator（计算 scratch buffer 大小并分配）
// 3. 获取计数器数据图像中的范围数量（numRanges）
// 4. 对每个指标名称：
//    a. 解析指标名称（移除修饰符如 '$', '&', '+'）
//    b. 转换为 MetricEvalRequest
//    c. 对每个范围：
//       - 获取范围描述符（如 "kernel_name"）
//       - 设置设备属性
//       - 调用 EvaluateToGpuValues 计算指标值
//       - 存储 (范围名称, 指标值) 对
// 5. 销毁 MetricsEvaluator
// 复杂度: O(M * R)，其中 M 是指标数量，R 是范围数量
```

```cpp
bool PrintMetricValues(
    std::string chipName,
    const std::vector<uint8_t>& counterDataImage,
    const std::vector<std::string>& metricNames,
    const uint8_t* pCounterAvailabilityImage = NULL);
// 格式化打印指标值到 stdout
// 输出格式: 表格形式，包含范围名称、指标名称、指标值三列，宽度分别为 40、100、自动
// 实现逻辑与 GetMetricGpuValue 类似，但直接输出而非返回数据结构
```

```cpp
std::vector<double> GetMetricValues(
    std::string chipName,
    const std::vector<uint8_t>& counterDataImage,
    const std::vector<std::string>& metricNames,
    const uint8_t* pCounterAvailabilityImage = NULL);
// 返回扁平化的指标值数组（遍历所有指标和所有范围）
// 与 GetMetricGpuValue 的区别：返回简单的 double 向量，不保留范围和指标的层次结构
// 适用场景: 仅需数值而不关心范围归属的情况
```

### 测量会话管理接口 (measureMetricPW.hpp)

```cpp
double measureMetricsStart(std::vector<std::string> newMetricNames);
// 初始化性能测量会话
// 执行步骤：
// 1. 延迟初始化 CUDA（调用 cudaFree(0) 激活 CUDA 运行时）
// 2. 获取 CUDA 设备并检查计算能力（要求 >= 7.0）
// 3. 初始化 CUPTI Profiler 和 NVPW Host
// 4. 查询芯片名称（如 "sm_80" for Ampere）
// 5. 获取计数器可用性图像
// 6. 调用 GetConfigImage 和 GetCounterDataPrefixImage 生成配置
// 7. 创建计数器数据图像（分配内存并初始化 scratch buffer）
// 8. 开始 CUPTI Profiler 会话（KernelReplay 模式，AutoRange 范围）
// 返回值: 成功返回 0.0，失败返回 -1.0 或 -2.0（不支持的设备）
```

```cpp
std::vector<double> measureMetricsStop();
// 结束性能测量会话并返回指标值
// 执行步骤：
// 1. 调用 runTestEnd() 结束 Profiler 会话（禁用测量、取消配置、结束会话）
// 2. 调用 GetMetricValues 提取指标值
// 3. 返回扁平化的指标值数组
// 注意: 不反初始化 CUPTI/NVPW（注释显示有意避免）
```

```cpp
void measureDRAMBytesStart();
// 预配置的 DRAM 带宽测量启动
// 测量指标: "dram__bytes_read.sum" 和 "dram__bytes_write.sum"
// 内部调用 measureMetricsStart 并传入固定的指标列表
```

```cpp
std::vector<double> measureDRAMBytesStop();
// 结束 DRAM 测量并返回值
// 直接返回 measureMetricsStop() 的结果，无额外处理
```

```cpp
void measureL2BytesStart();
// 预配置的 L2 缓存测量启动
// 测量指标: "lts__t_sectors_srcunit_tex_op_read.sum" 和 "lts__t_sectors_srcunit_tex_op_write.sum"
// 注意: 这些指标以 32-byte sector 为单位
```

```cpp
std::vector<double> measureL2BytesStop();
// 结束 L2 测量并返回值
// 后处理: 将返回的两个值乘以 32（从 sector 数转换为字节数）
// 算法: values[0] *= 32; values[1] *= 32
```

### Python 绑定接口 (pythonInterface.cpp)

```cpp
extern "C" PyObject* measureMetricStop();
// Python 可调用的测量停止函数
// 返回类型: Python list of float
// 执行流程：
// 1. 调用 runTestEnd() 结束 Profiler 会话
// 2. 调用 GetMetricValues 获取 C++ 向量
// 3. 获取 Python GIL (Global Interpreter Lock)
// 4. 创建 Python 列表并转换每个 double 为 PyFloat
// 5. 释放 GIL 并返回列表
// 注意: 名称与 measureMetricsStop 不同（缺少 's'），可能是遗留接口
```

### 辅助工具接口

```cpp
bool ParseMetricNameString(
    const std::string& metricName,
    std::string* reqName,
    bool* isolated,
    bool* keepInstances);
// 解析指标名称修饰符
// 支持的修饰符:
// - '$' 后缀: 孤立模式（isolated = true，默认）- 仅测量指定指标
// - '&' 后缀: 非孤立模式（isolated = false）- 允许与其他指标共享计数器
// - '+' 后缀: 保留实例（keepInstances = true）- 保留每个 GPU 实例的数据
// 处理步骤:
// 1. 移除 boost program_options 可能插入的 '\n' 字符
// 2. 移除尾部空格
// 3. 检测并移除修饰符，设置对应标志
// 4. 返回清理后的指标名称
```

```cpp
const char* GetNVPWResultString(NVPA_Status status);
// 将 NVPW 错误码转换为人类可读的错误消息
// 支持的错误类型包括:
// - NVPA_STATUS_ERROR: 通用错误
// - NVPA_STATUS_INVALID_ARGUMENT: 无效参数
// - NVPA_STATUS_OUT_OF_MEMORY: 内存不足
// - NVPA_STATUS_NOT_SUPPORTED: 操作不支持
// - NVPA_STATUS_UNSUPPORTED_GPU: GPU 不受支持
// 等 20+ 种错误状态
// 返回值: 静态字符串指针，包含错误名称
```

## 4. 使用示例

### 示例 1: 基本的 DRAM 带宽测量

```cpp
#include "cuda_metrics/measureMetricPW.hpp"

int main() {
    // 1. 开始测量 DRAM 读写字节
    measureDRAMBytesStart();

    // 2. 执行需要测量的 GPU 操作
    myKernel<<<grid, block>>>(...);  // CUDA kernel 启动
    cudaDeviceSynchronize();          // 等待 kernel 完成

    // 3. 停止测量并获取结果
    auto values = measureDRAMBytesStop();

    // 4. 分析结果
    double dramReadBytes = values[0];   // DRAM 读取字节数
    double dramWriteBytes = values[1];  // DRAM 写入字节数
    printf("DRAM Read: %.2f GB, Write: %.2f GB\n",
           dramReadBytes / 1e9, dramWriteBytes / 1e9);

    return 0;
}
```

### 示例 2: 自定义指标测量

```cpp
#include "cuda_metrics/measureMetricPW.hpp"

int main() {
    // 1. 定义感兴趣的指标列表
    std::vector<std::string> metrics = {
        "sm__pipe_tensor_cycles_active.sum",          // Tensor Core 激活周期
        "sm__pipe_tensor_cycles_active.avg.per_second", // Tensor Core 平均利用率
        "dram__throughput.avg.pct_of_peak_dram"      // DRAM 带宽利用率
    };

    // 2. 初始化测量会话
    measureMetricsStart(metrics);

    // 3. 执行测试代码
    runMyBenchmark();

    // 4. 获取结果
    auto results = measureMetricsStop();

    // 5. 处理结果（每个指标可能有多个范围的值）
    for (size_t i = 0; i < results.size(); ++i) {
        printf("Metric[%zu]: %.2f\n", i, results[i]);
    }

    return 0;
}
```

### 示例 3: 详细指标评估（保留范围信息）

```cpp
#include "cuda_metrics/Eval.hpp"
#include "cuda_metrics/Metric.hpp"
#include "cuda_metrics/measureMetricPW.hpp"

int main() {
    // 1. 准备配置
    std::vector<std::string> metrics = {"dram__bytes_read.sum"};
    std::string chipName = "sm_80";  // 假设使用 Ampere GPU

    // 2. 生成配置图像（离线配置）
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataPrefix;
    GetConfigImage(chipName, metrics, configImage, nullptr);
    GetCounterDataPrefixImage(chipName, metrics, counterDataPrefix, nullptr);

    // 3. 创建计数器数据图像
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> scratchBuffer;
    CreateCounterDataImage(counterDataImage, scratchBuffer, counterDataPrefix);

    // 4. 开始测量会话
    runTestStart(cuDevice, configImage, scratchBuffer,
                 counterDataImage, CUPTI_KernelReplay, CUPTI_AutoRange);

    // 5. 执行多个 kernel（自动范围划分）
    kernel1<<<...>>>(...);
    kernel2<<<...>>>(...);
    kernel3<<<...>>>(...);
    cudaDeviceSynchronize();

    // 6. 结束会话
    runTestEnd();

    // 7. 提取详细的带范围信息的指标值
    std::vector<NV::Metric::Eval::MetricNameValue> metricValues;
    GetMetricGpuValue(chipName, counterDataImage, metrics, metricValues, nullptr);

    // 8. 打印每个范围的结果
    for (const auto& mv : metricValues) {
        printf("Metric: %s\n", mv.metricName.c_str());
        for (const auto& rangeValue : mv.rangeNameMetricValueMap) {
            printf("  Range %s: %.2f bytes\n",
                   rangeValue.first.c_str(), rangeValue.second);
        }
    }

    return 0;
}
```

### 示例 4: Python 环境中的使用

```python
# 假设 pythonInterface.cpp 被编译为 Python 扩展模块
import cffi  # 或 ctypes、pybind11 等

# 加载编译好的共享库
metrics_lib = cffi.dlopen("./libcuda_metrics.so")

# 启动测量（通过 C++ 接口）
# 注意: Python 绑定仅提供了 measureMetricStop
# measureMetricsStart 需要通过 ctypes 调用

# 执行 GPU 操作
run_my_benchmark()

# 停止测量并获取结果
result_list = metrics_lib.measureMetricStop()
for value in result_list:
    print(f"Metric value: {value}")
```

## 5. 实现细节

### 内存管理

- **动态缓冲区分配策略**:
  - **Scratch Buffer**: 所有 NVPW 操作使用动态分配的 `std::vector<uint8_t>` 作为临时内存
    - 大小计算: 通过 `CalculateScratchBufferSize` API 查询所需字节数
    - 分配时机: 在每次评估器或配置对象初始化前
    - 生命周期: 与对应的 NVPW 对象生命周期绑定
  - **配置图像**: 使用两阶段分配模式
    - 第一次调用 `GetConfigImage` 且 `bytesAllocated=0, pBuffer=NULL` 获取所需大小
    - 第二次调用调整 `configImage.resize()` 并传入有效指针复制数据
  - **计数器数据图像**: 类似两阶段模式
    - 通过 `CalculateSize` 确定总大小
    - 分配主图像 + scratch buffer（两个独立向量）

- **RAII 资源管理**:
  - **ScopeExit 模式**: 使用 `SCOPE_EXIT` 宏确保 NVPW 对象在函数退出或异常时被销毁
    - 示例: `Metric.hpp` 第102行，`RawMetricsConfig` 的析构被注册为退出回调
    - 实现: 编译器生成唯一的匿名变量存储 `ScopeExit` 对象
    - 触发: 作用域结束时自动调用析构函数，进而执行 lambda
  - **避免资源泄漏**: 所有 NVPW 创建的对象（MetricsEvaluator, RawMetricsConfig, CounterDataBuilder）都有对应的 Destroy 调用

### 并发与线程安全

- **单线程假设**: 整个模块设计为单线程使用
  - 全局状态: 匿名命名空间中的 `cuContext`, `cuDevice`, `chipName` 等全局变量 (measureMetricPW.hpp 第61-71行)
  - 无互斥锁: 没有任何 `pthread_mutex` 或 `std::mutex` 的使用
  - Python GIL: `pythonInterface.cpp` 中显式调用 `PyGILState_Ensure` / `PyGILState_Release` 确保与 Python 解释器的交互安全

- **CUDA 上下文管理**:
  - **延迟初始化**: `measureMetricsStart` 中调用 `cudaFree(0)` 强制初始化 CUDA Runtime
  - **上下文获取**: 使用 `cuDeviceGet` 获取设备句柄，但注释显示 `cuCtxCreate` 被禁用 (measureMetricPW.hpp 第145行)
  - **默认上下文**: 依赖 CUDA Runtime 的隐式主上下文（Primary Context）

### 性能优化

- **批量指标处理**:
  - 一次 Profiler 会话可采集多个指标
  - `GetRawMetricRequests` 收集所有指标的原始依赖项（去重）
  - 减少 Profiler 采集通道数量，降低总测量开销

- **Kernel Replay 模式**:
  - 使用 `CUPTI_KernelReplay` 模式自动多次执行 kernel 以采集所有计数器
  - 优点: 对用户透明，无需手动多次启动 kernel
  - 缺点: 可能显著增加执行时间（取决于需要多少个采集通道）

- **评估器缓存**:
  - `GetMetricGpuValue` 在函数级别创建 MetricsEvaluator（每次调用新建）
  - 潜在优化: 如果多次评估相同指标，可缓存 evaluator 跨调用重用

### 错误处理

- **宏驱动的错误检查**:
  - **NVPW_API_CALL**: 检查 NVPA_Status，失败时打印错误并 `exit(-1)`
  - **CUPTI_API_CALL**: 检查 CUptiResult，失败时打印错误并 `exit(-1)`
  - **DRIVER_API_CALL**: 检查 CUresult，失败时打印错误并 `exit(-1)`
  - **RUNTIME_API_CALL**: 检查 cudaError_t，失败时打印错误并 `exit(-1)`
  - **RETURN_IF_NVPW_ERROR**: 更灵活的错误处理，允许指定返回值并打印 NVPW 错误字符串

- **错误传播策略**:
  - **致命错误**: 直接调用 `exit(-1)` 终止程序（如 CUDA API 失败）
  - **可恢复错误**: 返回 `false` 或 `nullptr`（如图像生成失败）
  - **错误信息**: 所有错误都输出到 `stderr`，包含文件名、行号和错误描述

- **特殊状态码**:
  - `measureMetricsStart` 返回 -1.0 表示配置失败
  - `measureMetricsStart` 返回 -2.0 表示设备计算能力 < 7.0（不支持 Profiler）

### 依赖关系

- **外部依赖**:
  - **CUDA Driver API**: `cuDeviceGet`, `cuDeviceGetAttribute`, `cuCtxCreate` (已禁用)
  - **CUDA Runtime API**: `cudaFree` (用于初始化)
  - **CUPTI Profiler API**:
    - `cuptiProfilerInitialize/DeInitialize`
    - `cuptiDeviceGetChipName`
    - `cuptiProfilerGetCounterAvailability`
    - `cuptiProfilerBeginSession/EndSession`
    - `cuptiProfilerSetConfig/UnsetConfig`
    - `cuptiProfilerEnableProfiling/DisableProfiling`
    - `cuptiProfilerCounterDataImage*` 系列函数
  - **NVPW (NVIDIA PerfWorks) Host API**:
    - `NVPW_InitializeHost`
    - `NVPW_CUDA_MetricsEvaluator_*` 系列
    - `NVPW_CUDA_RawMetricsConfig_*` 系列
    - `NVPW_CUDA_CounterDataBuilder_*` 系列
    - `NVPW_CounterData_*` 和 `NVPW_Profiler_CounterData_*` 系列
  - **Python C API**: `PyList_New`, `PyFloat_FromDouble`, `PyGILState_Ensure/Release` (仅 Python 绑定)

- **内部依赖关系**:
  - **Eval.hpp** → Parser.h, Utils.h, ScopeExit.h, nvperf_host.h, nvperf_cuda_host.h
  - **Metric.hpp** → Parser.h, Utils.h, nvperf_host.h, nvperf_cuda_host.h, ScopeExit.h
  - **measureMetricPW.hpp** → Eval.hpp, Metric.hpp, CUDA/CUPTI 头文件
  - **pythonInterface.cpp** → measureMetricPW.hpp, Python.h

### 设计模式

- **RAII (Resource Acquisition Is Initialization)**:
  - `ScopeExit<T>` 模板类确保资源在作用域结束时释放
  - 所有 NVPW 对象的销毁都通过显式 API 调用，但使用 `SCOPE_EXIT` 自动化

- **工厂模式**:
  - `NVPW_CUDA_MetricsEvaluator_Initialize`, `NVPW_CUDA_RawMetricsConfig_Create_V2` 等函数充当工厂，创建不透明的 NVPW 对象句柄

- **两阶段初始化模式**:
  - 用于所有缓冲区分配：先查询大小，再分配并填充数据
  - 示例: `GetConfigImage` (Metric.hpp 第122-131行)

- **策略模式**:
  - Profiler 支持不同的重放模式 (`CUpti_ProfilerReplayMode`): KernelReplay, UserReplay
  - 支持不同的范围模式 (`CUpti_ProfilerRange`): AutoRange, UserRange
  - 当前实现固定使用 KernelReplay + AutoRange

- **命名空间层次**:
  - `NV::Metric::Config`: 配置生成相关函数
  - `NV::Metric::Eval`: 指标评估相关函数
  - `NV::Metric::Parser`: 解析工具
  - `NV::Metric::Utils`: 错误处理工具
  - 匿名命名空间 (`measureMetricPW.hpp` 第60-72行): 隐藏全局状态

### 算法复杂度

- **GetRawMetricRequests**:
  - 时间复杂度: O(M)，其中 M 是指标数量
  - 空间复杂度: O(D)，其中 D 是原始依赖项数量（通常 D >= M）

- **GetConfigImage**:
  - 时间复杂度: O(M + R)，其中 M 是指标数量，R 是原始指标请求数量
  - 空间复杂度: O(S)，其中 S 是配置图像大小（由 NVPW 内部计算）

- **GetMetricGpuValue**:
  - 时间复杂度: O(M * R * E)，其中 M 是指标数，R 是范围数，E 是评估单个指标的时间
  - 空间复杂度: O(S + M*R)，S 是 scratch buffer 大小，M*R 是结果存储

- **CreateCounterDataImage**:
  - 时间复杂度: O(1)（仅分配和初始化）
  - 空间复杂度: O(C + S)，其中 C 是计数器数据图像大小，S 是 scratch buffer 大小

### 关键常量和配置

- **设备要求**: 计算能力 >= 7.0 (Volta 及更新架构) (measureMetricPW.hpp 第210行)
- **默认范围数量**: `numRanges = 2` (measureMetricPW.hpp 第56行)
- **最大范围树节点**: `maxNumRangeTreeNodes = numRanges` (measureMetricPW.hpp 第82行)
- **最大范围名称长度**: `maxRangeNameLength = 64` (measureMetricPW.hpp 第83行)
- **Profiler 模式**:
  - 重放模式: `CUPTI_KernelReplay` (自动重放 kernel) (measureMetricPW.hpp 第269行)
  - 范围模式: `CUPTI_AutoRange` (自动划分范围) (measureMetricPW.hpp 第270行)

### 已知限制和注意事项

1. **全局状态管理**: 匿名命名空间中的全局变量使得模块不是线程安全的，也不能同时运行多个测量会话

2. **Python GIL 依赖**: Python 绑定假设调用者已持有 GIL 或通过 `PyGILState_Ensure` 获取

3. **CUDA 初始化**: 依赖 `cudaFree(0)` 的副作用来初始化 CUDA Runtime，这是一种 hack

4. **计算能力检查**: 硬编码要求计算能力 >= 7.0，不支持 Pascal (6.x) 及更早架构

5. **内存开销**: Kernel Replay 模式可能导致显着的内存占用（取决于 scratch buffer 大小）

6. **性能影响**: Profiling 会显著降低 kernel 执行性能（可能 10x-100x 慢），不适合生产环境

7. **错误恢复**: 大部分错误直接调用 `exit(-1)`，无法优雅恢复或清理资源

8. **计数器数据前缀**: 假设所有指标的计数器数据前缀是相同的（在 `GetCounterDataPrefixImage` 中未区分指标）
