# `infini_run` 分布式训练启动器核心实现文档

这是一个轻量级的分布式训练进程管理工具，用于在单节点或多节点环境下启动和管理多个训练进程。通过 fork 方式并行启动训练进程，并自动配置分布式训练所需的环境变量。

## 1. 模块结构

- **`infini_run.cc`**: 核心实现文件，包含进程管理逻辑和环境变量配置
- **`CMakeLists.txt`**: 构建配置，链接 gflags 和 glog 库

## 2. 核心类与函数

### `main` 函数
- **Location**: `infini_run.cc:16-58`
- **Primary Function**: 分布式训练启动器的入口点，负责参数解析、进程 fork 和环境变量配置
- **Algorithm**:
  1. 解析命令行参数和训练程序路径
  2. 计算 world_size（总进程数）和解析 rendezvous endpoint
  3. 循环 fork 出 N 个子进程（N = nproc_per_node）
  4. 每个子进程设置独立的环境变量后执行训练程序
  5. 父进程等待所有子进程结束

## 3. API 接口

### 命令行接口

```bash
infini_run [OPTIONS] <train_program> [train_args...]

# 必需参数
train_program: 要执行的训练程序路径
train_args: 传递给训练程序的额外参数

# 可选标志
--nnodes=int              # 总节点数 (default: 1)
--nproc_per_node=int      # 每节点进程数 (default: 1)
--node_rank=int           # 当前节点排名 (default: 0)
--rdzv_endpoint=string    # Rendezvous 端点 "host:port" (default: "127.0.0.1:29500")
```

### 环境变量输出

每个训练进程会收到以下环境变量：

```cpp
NNODES=<int>              // 总节点数
NPROC_PER_NODE=<int>      // 每节点进程数
MASTER_ADDR=<string>      // 主节点地址（从 rdzv_endpoint 提取）
MASTER_PORT=<string>      // 主节点端口（从 rdzv_endpoint 提取）
GLOBAL_PROC_RANK=<int>    // 全局进程排名（0 到 world_size-1）
LOCAL_PROC_RANK=<int>     // 本地进程排名（0 到 nproc_per_node-1）
PROC_WORLD_SIZE=<int>     // 全局总进程数（nnodes * nproc_per_node）
```

## 4. 使用示例

### 单节点多进程训练

```bash
# 在单个节点上启动 8 个训练进程
./infini_run --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --rdzv_endpoint=127.0.0.1:29500 \
    /path/to/training_program --batch_size=32 --lr=0.001

# 每个进程会获得：
# - NNODES=1, NPROC_PER_NODE=8, PROC_WORLD_SIZE=8
# - GLOBAL_PROC_RANK=0..7, LOCAL_PROC_RANK=0..7
# - MASTER_ADDR=127.0.0.1, MASTER_PORT=29500
```

### 多节点分布式训练

```bash
# 节点 0 (node_rank=0):
./infini_run --nnodes=4 --nproc_per_node=8 --node_rank=0 \
    --rdzv_endpoint=192.168.1.100:29500 \
    /path/to/training_program --batch_size=32

# 节点 1 (node_rank=1):
./infini_run --nnodes=4 --nproc_per_node=8 --node_rank=1 \
    --rdzv_endpoint=192.168.1.100:29500 \
    /path/to/training_program --batch_size=32

# 每个节点会启动 8 个进程，总共 32 个进程
# 节点 0 的进程: GLOBAL_PROC_RANK 0-7
# 节点 1 的进程: GLOBAL_PROC_RANK 8-15
# 节点 2 的进程: GLOBAL_PROC_RANK 16-23
# 节点 3 的进程: GLOBAL_PROC_RANK 24-31
```

### 环境变量在训练程序中的使用

```cpp
// 训练程序可以使用这些环境变量初始化分布式后端
#include <cstdlib>
#include <iostream>

void init_distributed() {
    int world_size = std::atoi(std::getenv("PROC_WORLD_SIZE"));
    int rank = std::atoi(std::getenv("GLOBAL_PROC_RANK"));
    std::string master_addr = std::getenv("MASTER_ADDR");
    int master_port = std::atoi(std::getenv("MASTER_PORT"));

    // 使用这些信息初始化通信后端（如 NCCL、Gloo 等）
    std::cout << "Initializing rank " << rank
              << " of " << world_size
              << " at " << master_addr << ":" << master_port << std::endl;
}
```

## 5. 实现细节

### 进程管理策略

- **Fork 模型**: 使用 POSIX `fork()` 创建子进程，每个进程独立运行训练程序
- **同步等待**: 父进程通过 `wait()` 循环等待所有子进程结束，确保所有训练进程完成
- **进程隔离**: 每个子进程通过 `execvp()` 替换为训练程序，完全独立运行
- **错误处理**: 如果 `execvp()` 失败，子进程会输出错误信息并退出（line 47-48）

### 环境变量配置算法

1. **地址解析** (line 28-29):
   - 从 `rdzv_endpoint` 字符串中提取主机地址和端口
   - 使用 `substr()` 和 `find(':')` 分割 "host:port" 格式

2. **全局排名计算** (line 34):
   ```cpp
   global_proc_rank = node_rank * nproc_per_node + local_proc_rank
   ```
   - 保证每个进程有唯一的全局 ID（0 到 world_size-1）
   - 同一节点内的进程 GLOBAL_PROC_RANK 连续

3. **环境变量设置** (line 35-44):
   - 使用 `setenv()` 设置环境变量（overwrite=1）
   - 所有变量都以字符串形式传递（通过 `std::to_string()` 转换）

### 依赖库

- **gflags** (Google Commandline Flags): 命令行参数解析
- **glog** (Google Logging): 日志记录和 CHECK 宏
- **POSIX API**: `fork()`, `execvp()`, `wait()`, `setenv()`

### 设计模式

- **Process Pool Pattern**: 预先创建固定数量的子进程（nproc_per_node）
- **Leader-Follower**: 父进程作为 leader，等待所有 follower（子进程）完成
- **Environment Injection**: 通过环境变量传递配置信息，而非命令行参数

### 性能特征

- **时间复杂度**: O(nproc_per_node) - fork 和 wait 操作与进程数线性相关
- **空间复杂度**: O(nproc_per_node) - 每个子进程独立占用内存空间
- **并发模型**: 真并行（多进程充分利用多核 CPU）

### 错误处理

- **参数验证**: 使用 `CHECK_GE(argc, 2)` 确保提供了训练程序路径（line 20）
- **执行失败**: 如果 `execvp()` 失败，调用 `perror()` 打印错误并退出（line 47-48）
- **进程回收**: 父进程通过循环 `wait()` 确保所有子进程被回收，避免僵尸进程（line 52-55）

### 典型使用场景

1. **单机多卡训练**: `nproc_per_node=8` 在 8 卡服务器上启动 8 个进程
2. **多机多卡训练**: `nnodes=4, nproc_per_node=8` 在 4 台机器上启动 32 个进程
3. **参数服务器**: 通过 rdzv_endpoint 配置中心协调节点地址
4. **弹性训练**: 结合进程排名实现故障恢复和检查点保存

### 限制与注意事项

- 仅支持 Unix-like 系统（依赖 POSIX API）
- 不支持 Windows 系统
- 父进程阻塞等待，不提供异步监控接口
- 不处理子进程崩溃后的自动重启
- 进程间通信需要训练程序自行实现（如通过 PyTorch DDP、DeepSpeed 等）
