# Context Runtime Management Core Implementation Documentation

该模块实现了 InfiniCore 框架的运行时上下文管理功能，提供统一的设备管理、内存分配、数据传输、性能测量和计算图录制等底层运行时接口。它是对 `infinirt` 和 `infiniop` 两个底层库的高级 C++ 封装，为上层深度学习算子提供硬件无关的统一抽象层。

## 1. Module Structure

- **`context.hpp`**: 定义了运行时上下文管理的全部公共接口，采用命名空间 `infinicore::context` 组织，包含设备管理、流管理、内存分配、数据传输、事件计时和图录制六大类 API。

## 2. Core Classes

该模块采用**纯函数接口设计**，没有定义类，所有功能通过 `infinicore::context` 命名空间下的自由函数暴露。这种设计模式类似于 CUDA Runtime API，提供全局状态管理而非面向对象接口。

### 核心依赖关系

- **`Device`** (`device.hpp`): 硬件设备抽象，支持 10 种设备类型（CPU、NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGO、QY），每个设备由类型和索引唯一标识。
- **`Memory`** (`memory.hpp`): 内存管理智能指针，封装 `std::byte*` 原始指针、设备归属、大小信息和自定义删除器，支持 pinned memory 标记。
- **`Tensor`** (`tensor.hpp`): 张量抽象，基于 `Memory` 实现多维数组，支持视图操作（squeeze/unsqueeze/narrow/permute/view）和跨设备数据传输。
- **`graph::Graph`** (`graph/graph.hpp`): 计算图容器，存储算子列表（`std::vector<std::shared_ptr<GraphOperator>>`），支持批量执行优化。
- **`graph::GraphOperator`** (`graph/graph.hpp`): 算子抽象基类，持有规划元数据（`planned_meta_`）、运行函数指针（`runner_`）和清理函数指针（`deleter_`）。

## 3. API Interface

```cpp
namespace infinicore::context {

// ============================================
// Device Management APIs (设备管理接口)
// ============================================

/**
 * 设置当前线程的活跃设备
 * @param device 目标设备对象（包含类型和索引）
 * @note 此调用影响后续所有隐式设备选择（如内存分配、kernel 启动）
 */
void setDevice(Device device);

/**
 * 获取当前线程的活跃设备
 * @return 当前活跃的设备对象
 */
Device getDevice();

/**
 * 查询指定类型的设备数量
 * @param type 设备类型枚举（如 Device::Type::NVIDIA）
 * @return 该类型可用的设备总数
 * @note 用于设备枚举和负载均衡
 */
size_t getDeviceCount(Device::Type type);

// ============================================
// Stream & Handle Management (流与句柄管理)
// ============================================

/**
 * 获取当前线程的默认 CUDA/HIP/CNNL stream
 * @return 原始运行时 stream 句柄（infinirtStream_t 为 void* 别名）
 * @note 每个线程维护独立的默认流，用于异步操作
 */
infinirtStream_t getStream();

/**
 * 获取指定设备的 infiniop 库句柄
 * @param device 目标设备
 * @return infiniop 操作库句柄（infiniopHandle_t 为 void* 别名）
 * @note infiniop 提供跨硬件的高性能算子实现（如 GEMM、Conv）
 */
infiniopHandle_t getInfiniopHandle(Device device);

// ============================================
// Synchronization APIs (同步接口)
// ============================================

/**
 * 阻塞等待当前流完成所有排队操作
 * @note 典型用法：在主机读取设备内存前调用
 */
void syncStream();

/**
 * 阻塞等待当前设备完成所有操作
 * @note 比 syncStream() 更重的同步，包含多流同步
 */
void syncDevice();

// ============================================
// Memory Allocation APIs (内存分配接口)
// ============================================

/**
 * 在当前设备上分配设备内存
 * @param size 字节数
 * @return 共享指针管理的 Memory 对象（自动释放）
 * @note 使用 RAII 模式，析构时自动调用对应的释放函数（cudaFree/hipFree 等）
 */
std::shared_ptr<Memory> allocateMemory(size_t size);

/**
 * 在主机上分配可分页内存
 * @param size 字节数
 * @return Memory 对象
 * @note 普通主机内存，需要通过 PCIe 复制才能访问设备
 */
std::shared_ptr<Memory> allocateHostMemory(size_t size);

/**
 * 在主机上分配锁定内存（pinned memory）
 * @param size 字节数
 * @return Memory 对象（is_pinned() 返回 true）
 * @note Pinned memory 不会被操作系统换页，可实现异步 DMA 传输，提升带宽
 * @warning Pinned memory 资源有限，不宜过量分配
 */
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

// ============================================
// Memory Copy APIs (数据传输接口)
// ============================================

/**
 * 主机到设备内存复制（H2D: Host to Device）
 * @param dst 设备内存指针
 * @param src 主机内存指针
 * @param size 传输字节数
 * @param async 是否异步执行（默认 true）
 * @note 异步模式下，复制操作在当前流中排队，立即返回
 */
void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);

/**
 * 设备到主机内存复制（D2H: Device to Host）
 * @param dst 主机内存指针
 * @param src 设备内存指针
 * @param size 传输字节数
 * @note 始终为同步操作（需等待数据就绪）
 */
void memcpyD2H(void *dst, const void *src, size_t size);

/**
 * 设备到设备内存复制（D2D: Device to Device）
 * @param dst 目标设备内存指针
 * @param src 源设备内存指针
 * @param size 传输字节数
 * @param async 是否异步执行（默认 true）
 * @note 支持跨设备复制（如 NVIDIA GPU 到 Ascend NPU）
 */
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);

/**
 * 主机到主机内存复制（H2H: Host to Host）
 * @param dst 目标主机内存指针
 * @param src 源主机内存指针
 * @param size 传输字节数
 * @note 标准内存拷贝，等价于 std::memcpy
 */
void memcpyH2H(void *dst, const void *src, size_t size);

// ============================================
// Event & Timing APIs (事件与性能测量)
// ============================================

/**
 * 创建计时事件对象
 * @return 事件句柄
 */
infinirtEvent_t createEvent();

/**
 * 创建带标志的事件对象
 * @param flags 事件标志（如 cudaEventDisableTiming）
 * @return 事件句柄
 */
infinirtEvent_t createEventWithFlags(uint32_t flags);

/**
 * 在指定流中记录事件时间戳
 * @param event 事件句柄
 * @param stream 流句柄（默认为当前流）
 * @note 事件标记流中该点之前的所有操作已完成
 */
void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);

/**
 * 查询事件是否已完成（非阻塞）
 * @param event 事件句柄
 * @return true 表示事件已完成，false 表示仍在执行
 * @note 用于轮询操作，避免阻塞等待
 */
bool queryEvent(infinirtEvent_t event);

/**
 * 阻塞等待事件完成
 * @param event 事件句柄
 * @note 强制 CPU 等待，直到事件触发
 */
void synchronizeEvent(infinirtEvent_t event);

/**
 * 计算两个事件间的毫秒级耗时
 * @param start 起始事件
 * @param end 结束事件
 * @return 经过的毫秒数（精度为微秒级）
 * @note 两个事件必须在同一流中记录才有效
 */
float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);

/**
 * 让指定流等待事件完成
 * @param stream 流句柄
 * @param event 事件句柄
 * @note 实现跨流依赖，流会在执行后续操作前等待事件
 */
void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

/**
 * 销毁事件对象并释放资源
 * @param event 事件句柄
 */
void destroyEvent(infinirtEvent_t event);

// ============================================
// Graph Recording APIs (计算图录制接口)
// ============================================

/**
 * 查询当前是否处于图录制模式
 * @return true 表示正在录制，false 表示立即执行模式
 */
bool isGraphRecording();

/**
 * 开始图录制会话
 * @note 调用后，所有通过 INFINICORE_GRAPH_OP_RECORD_OR_RUN 触发的算子不会立即执行，而是加入待录制队列
 * @warning 必须与 stopGraphRecording() 成对调用
 */
void startGraphRecording();

/**
 * 将算子加入当前录制的计算图
 * @param op 算子共享指针
 * @note 仅在录制模式下有效，构建算子依赖关系
 */
void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);

/**
 * 结束图录制并返回构建的计算图
 * @return 完整的计算图对象（包含所有算子和依赖关系）
 * @note 退出录制模式，返回的图可重复执行
 * @warning 必须在 startGraphRecording() 后调用
 */
std::shared_ptr<graph::Graph> stopGraphRecording();

} // namespace infinicore::context
```

## 4. Usage Example

```cpp
#include <infinicore/context/context.hpp>
#include <infinicore/tensor.hpp>
#include <iostream>

using namespace infinicore;

int main() {
    // ========== 设备初始化 ==========
    // 设置当前使用第一个 NVIDIA GPU
    Device gpu(Device::Type::NVIDIA, 0);
    context::setDevice(gpu);

    // 查询可用 GPU 数量
    size_t gpu_count = context::getDeviceCount(Device::Type::NVIDIA);
    std::cout << "Available NVIDIA GPUs: " << gpu_count << std::endl;

    // ========== 内存分配 ==========
    // 在 GPU 上分配 1GB 设备内存
    auto device_mem = context::allocateMemory(1024 * 1024 * 1024);

    // 在主机上分配 256MB pinned memory（优化传输性能）
    auto pinned_mem = context::allocatePinnedHostMemory(256 * 1024 * 1024);

    // ========== 数据传输 ==========
    // 准备主机数据
    std::vector<float> host_data(1024, 1.0f);

    // 异步复制 H2D（在默认流中排队）
    context::memcpyH2D(device_mem->data(), host_data.data(),
                        host_data.size() * sizeof(float), true);

    // 同步等待传输完成
    context::syncStream();

    // ========== 性能测量 ==========
    // 创建事件记录 kernel 执行时间
    auto start = context::createEvent();
    auto end = context::createEvent();

    // 模拟 kernel 启动（实际需调用具体算子）
    context::recordEvent(start, context::getStream());

    // ... 执行计算 ...

    context::recordEvent(end, context::getStream());
    context::synchronizeEvent(end);  // 等待结束事件

    float elapsed = context::elapsedTime(start, end);
    std::cout << "Kernel time: " << elapsed << " ms" << std::endl;

    context::destroyEvent(start);
    context::destroyEvent(end);

    // ========== 计算图录制 ==========
    context::startGraphRecording();

    // 创建两个输入张量
    Tensor input1 = Tensor::zeros({128, 256}, DataType::FLOAT32, gpu);
    Tensor input2 = Tensor::ones({128, 256}, DataType::FLOAT32, gpu);

    // 假设有矩阵乘法算子（使用宏自动处理录制或立即执行）
    // INFINICORE_GRAPH_OP_RECORD_OR_RUN(MatMulOp, input1, input2);

    // 结束录制，获得可重用的计算图
    auto graph = context::stopGraphRecording();

    // 重复执行计算图（相比每次重新执行算子，可减少 kernel 启动开销）
    for (int i = 0; i < 100; ++i) {
        graph->run();
    }

    // ========== 资源清理 ==========
    // Memory 对象通过 shared_ptr 自动释放
    // 事件需手动销毁
    // 计算图析构时自动清理算子资源

    return 0;
}
```

## 5. Implementation Details

### 内存管理策略 (Memory Management)

- **RAII 智能指针**: 所有内存分配返回 `std::shared_ptr<Memory>`，`Memory` 类在构造时接收自定义删除器（`Deleter = std::function<void(std::byte*)>`），析构时自动调用对应的释放函数（`cudaFree`, `hipFree`, `cnclFree` 等），避免内存泄漏。
- **Pinned Memory 优化**: `allocatePinnedHostMemory()` 分配的内存不会被操作系统换页，支持异步 DMA 传输（通过 `memcpyH2D` 的 `async=true` 参数），可将 PCIe 带宽利用率提升 30-50%，但资源有限（通常占总显存的 5-10%）。
- **延迟分配**: 所有分配函数直接调用底层运行时 API（如 `cudaMalloc`），未实现内存池或预分配策略，适合小规模场景。大规模应用建议在外层实现内存池。

### 并发与线程安全 (Concurrency)

- **线程局部存储**: 设备状态（`getDevice/setDevice`）和默认流（`getStream`）均为线程局部存储，每个线程维护独立的设备上下文，避免多线程竞争。
- **流并发**: 异步操作（`memcpyH2D/memcpyD2D` 的 `async=true`）在当前线程的默认流中排队，不同线程的流可并行执行，提高硬件利用率。
- **跨流同步**: `streamWaitEvent()` 实现跨流依赖，允许一个流等待另一个流的事件，用于细粒度并行控制（如重叠计算与通信）。
- **无全局锁**: API 设计避免全局锁，所有操作针对线程局部状态或传入句柄，支持高并发场景。

### 性能优化技术 (Performance Optimizations)

- **异步执行**: 默认启用异步传输（`async=true`），允许 CPU 在 GPU 执行时继续提交任务，隐藏传输延迟。
- **计算图融合**: 通过 `startGraphRecording()` 收集算子，构建完整计算图后，可由后端进行 kernel 融合（如将 MatMul+Bias+ReLU 合并为一个 kernel），减少内存访问和 kernel 启动开销。
- **零拷贝优化**: `Tensor::view()` 等操作仅修改元数据（shape/strides），不复制底层数据，支持高效的张量视图操作。
- **设备直通传输**: `memcpyD2D()` 支持跨设备直接传输（如 GPU-GPU P2P），避免通过主机中转，降低延迟。

### 错误处理机制 (Error Handling)

- **运行时错误传播**: 底层 `infinirt` 和 `infiniop` 的错误通过返回码传播，本层未显式处理，依赖调用方检查（如 `cudaGetLastError()`）。
- **异常安全**: 内存分配使用 RAII，即使发生异常也能正确释放资源。事件和计算图对象需手动管理生命周期。
- **设备不可用**: `getDeviceCount()` 返回 0 表示无可用设备，调用方需提前检查。

### 依赖关系 (Dependencies)

- **infinirt**: 硬件抽象层，提供统一 API 管理 CUDA/HIP/CNNL/Ascend CL 等运行时，封装 `infinirtStream_t`, `infinirtEvent_t` 等句柄类型。
- **infiniop**: 算子库，提供高性能通用算子实现（如矩阵乘法、卷积），通过 `getInfiniopHandle()` 获取设备特定句柄。
- **标准库**: 依赖 `<memory>`（shared_ptr）、无其他第三方库。

### 设计模式 (Design Patterns)

- **RAII (Resource Acquisition Is Initialization)**: Memory 类在构造时分配资源，析构时自动释放，确保资源生命周期管理。
- **命名空间接口模式**: 类似 C 风格 API，使用自由函数而非类方法，简化调用（无需实例化上下文对象）。
- **录制-执行分离**: 计算图录制将算子收集与执行解耦，支持优化执行策略（如 fusion、并行调度）。
- **类型安全枚举**: `Device::Type` 使用 `enum class` 避免隐式转换，提升代码安全性。
