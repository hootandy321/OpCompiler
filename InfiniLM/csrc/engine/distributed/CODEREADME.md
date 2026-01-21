# Distributed Engine Core Implementation Documentation

分布式训练引擎通信管理模块，负责张量并行（Tensor Parallelism）环境下的设备通信组初始化、配置管理和资源生命周期维护。该模块基于 InfiniCCL 通信库构建，提供多设备间的高性能通信抽象。

## 1. Module Structure

- **`dist_config.hpp/cpp`**: 分布式配置数据结构，定义张量并行的设备 ID 映射关系
- **`communication_group.hpp/cpp`**: 通信组管理器，封装 InfiniCCL 通信域的初始化与销毁
- **`distributed.hpp`**: 模块统一头文件入口，聚合分布式功能接口

## 2. Core Classes

### `DistConfig`
- **Location**: `dist_config.hpp/cpp`
- **Primary Function**: 张量并行设备拓扑配置的纯数据容器，描述如何将逻辑 rank 映射到物理设备 ID
- **Key Members**:
  - `tp_device_ids: std::vector<int>`: 每个 tensor parallel rank 对应的物理设备 ID 列表，索引即 rank 编号
- **Core Methods**:
  - `DistConfig()`: 默认构造单设备配置 (tp_device_ids={0})
  - `DistConfig(int tp_size)`: 构造连续设备 ID 配置，生成 [0, 1, ..., tp_size-1] 序列，时间复杂度 O(tp_size)
  - `DistConfig(const std::vector<int>& tp_device_ids_)`: 从已有设备 ID 列表构造，支持自定义非连续映射
  - `operator std::string()`: 生成可读配置字符串，格式 "DistConfig(tp_device_ids=[0, 1, 2])"
- **Lifecycle**: 值类型语义，支持拷贝和移动，无动态资源管理

### `CommunicationGroup`
- **Location**: `communication_group.hpp/cpp`
- **Primary Function**: 管理多设备间 InfiniCCL 通信域的生命周期，提供 rank 信息查询接口
- **Key Members**:
  - `dist_config_: DistConfig`: 设备拓扑配置的副本，确保配置独立性
  - `device_type_: infinicore::Device::Type`: 目标设备类型 (CUDA/ROCm/BANG 等)
  - `communicators_: std::vector<infinicclComm_t>`: 每个 rank 对应的 InfiniCCL 通信句柄，按 rank 索引存储
- **Core Methods**:
  - `CommunicationGroup(const DistConfig&, Device::Type)`: 构造时根据 tp_device_ids 初始化通信域，自动切换设备上下文，调用 `infinicclCommInitAll` 批量创建通信域（仅当 tp_size > 1 时）
  - `get_rank_info(int rank)`: 返回指定 rank 的完整元信息（设备对象、tp_size、tp_rank、通信句柄），通过索引查找 O(1)
  - `get_world_size()`: 返回总进程数（即 tp_device_ids.size()），O(1)
  - `get_dist_config()`: 获取内部配置的常量引用
  - `~CommunicationGroup()`: 析构时遍历销毁所有通信域，调用 `infinicclCommDestroy`，避免资源泄漏
- **Lifecycle**: RAII 管理模式，构造时批量初始化通信域，析构时确保清理

### `RankInfo`
- **Location**: `communication_group.hpp`
- **Primary Function**: 单个 rank 的运行时元数据封装，聚合设备信息、并行拓扑和通信句柄
- **Key Members**:
  - `device: infinicore::Device`: 绑定的物理设备对象（设备类型 + ID）
  - `tp_size: int`: 张量并行总规模
  - `tp_rank: int`: 当前 rank 在张量并行组内的编号
  - `comm: infinicclComm_t`: InfiniCCL 通信句柄
- **Core Methods**:
  - `RankInfo(Device)`: 构造函数，默认从全局上下文获取当前设备，初始化 tp_size=1, tp_rank=0, comm=nullptr
  - `to_string()`: 生成人类可读的描述字符串，包含设备类型、tp_size、tp_rank
- **Lifecycle**: 值类型，由 `CommunicationGroup::get_rank_info()` 按值返回

## 3. API Interface

```cpp
// 配置张量并行拓扑
DistConfig config(4);  // 4卡张量并行，设备ID [0,1,2,3]
DistConfig custom_config({0, 2, 4, 6});  // 自定义非连续设备映射

// 初始化通信组
CommunicationGroup group(config, infinicore::Device::Type::CUDA);

// 查询运行时信息
int world_size = group.get_world_size();  // 总进程数
RankInfo rank0 = group.get_rank_info(0);  // rank 0 的元信息
rank0.comm;  // InfiniCCL 通信句柄，用于集合通信

// 访问配置
const DistConfig& config_ref = group.get_dist_config();
```

## 4. Usage Example

```cpp
// 场景：在 8 卡机器上启动 4 卡张量并行的推理服务
#include "distributed.hpp"

using namespace infinilm::engine::distributed;

// 步骤 1: 配置张量并行拓扑
// 方案A - 使用前4张卡
DistConfig config(4);
// 方案B - 指定卡号 (如使用卡 0,2,4,6 避免单点过热)
DistConfig config({0, 2, 4, 6});

// 步骤 2: 初始化通信组 (自动创建 InfiniCCL 通信域)
CommunicationGroup comm_group(config, infinicore::Device::Type::CUDA);

// 步骤 3: 获取各 rank 的设备信息和通信句柄
for (int rank = 0; rank < comm_group.get_world_size(); ++rank) {
    RankInfo info = comm_group.get_rank_info(rank);

    // 设置当前上下文到目标设备
    infinicore::context::setDevice(info.device);

    // 使用 info.comm 执行 AllReduce/AllGather 等集合通信
    // infinicclAllReduce(..., info.comm, ...);
}

// 步骤 4: 生命周期结束自动清理 (RAII)
// 析构函数自动调用 infinicclCommDestroy 释放所有通信域
```

## 5. Implementation Details

- **内存管理**: 零堆分配设计，`DistConfig` 使用 `std::vector<int>` 存储设备 ID，`CommunicationGroup` 使用 `std::vector<infinicclComm_t>` 存储通信句柄，均依赖 STL 容器的自动内存管理
- **并发控制**: 无显式锁机制，假设在单线程环境或外部同步下调用；InfiniCCL 内部通信句柄是线程安全的
- **性能优化**:
  - 使用 `infinicclCommInitAll` 批量初始化多通信域，减少初始化开销
  - `get_rank_info` 返回值避免内部状态外泄，保证线程安全
  - 配置对象按值存储，避免共享状态的竞态条件
- **错误处理**: 通过 `RUN_INFINI` 宏包装 InfiniCCL 调用，将错误码转换为异常（定义在 `../../utils.hpp`），确保通信初始化失败时快速失败
- **依赖关系**:
  - **外部依赖**: `infiniccl.h` (通信库), `infinicore/context/context.hpp` (设备管理)
  - **内部依赖**: `../../utils.hpp` (RUN_INFINI 宏)
- **设计模式**:
  - **RAII (Resource Acquisition Is Initialization)**: `CommunicationGroup` 构造时获取通信资源，析构时释放，杜绝资源泄漏
  - **Factory Method**: `DistConfig` 多个构造函数提供不同配置策略
  - **Immutable Configuration**: `DistConfig` 创建后不可变，保证配置在通信组生命周期内稳定

- **设备上下文切换策略**: 构造函数检查当前设备类型与目标类型，不匹配时调用 `infinicore::context::setDevice()` 切换，确保后续 InfiniCCL 操作在正确设备上执行
- **通信域生命周期**: 仅当 `tp_size > 1` 时才创建通信域（单卡场景跳过初始化），避免不必要的资源开销；析构时同样检查 size，防止空指针访问
- **设备 ID 映射灵活性**: `DistConfig` 支持非连续、任意顺序的设备 ID 映射，适应异构拓扑（如跨 NUMA 节点、避免 PCIe 竞争等场景）
