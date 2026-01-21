# SYCL GPU L2 Cache Benchmark 核心实现文档

本模块是一个基于 SYCL (SYCL-compatible with NVIDIA CUDA) 的 GPU L2 缓存性能测试工具，通过控制内存访问步长来探测 GPU L2 缓存的大小和带宽特性。

## 1. 模块结构

- **`sycl-gpu-l2-cache.cpp`**: 核心基准测试程序，实现不同内存跨度下的 GPU 内存访问性能测试
- **`build.sh`**: 编译脚本，使用 clang++ 的 SYCL 后端针对 NVIDIA GPU (sm_80 架构) 进行编译

## 2. 核心实现

### 主程序 `main()`
- **Location**: `sycl-gpu-l2-cache.cpp:13-74`
- **Primary Function**: 通过指数增长的数据集大小，测试不同内存跨度下的 GPU 内存访问带宽，从而识别 L2 缓存的容量边界和性能特征
- **Key Constants**:
  - `N = 64`: 内部循环迭代次数，控制每个工作项的内存访问次数
  - `blockSize = 1024`: SYCL work-group 大小（对应 CUDA 的线程块大小）
  - `blockCount = 200000`: 并行执行的 work-group 数量

### 核心测试循环
- **Outer Loop** (`line 24`): 控制内存跨度 `blockRun`，从 3 开始按指数增长（每次增加 10%），覆盖从几十 KB 到数百 MB 的数据集范围
- **Inner Loop** (`line 30-60`): 对每个内存跨度进行 11 次重复测试，通过微调缓冲区大小（`i * 128`）来避免缓存预热效应

### 内存访问模式
- **Allocation** (`line 32-33`):
  - 使用 `malloc_device<dtype>()` 在设备端分配两个缓冲区 `dA` 和 `dB`
  - 缓冲区大小为 `blockRun * blockSize * N + i * 128`，随外层循环指数增长

- **Initialization Kernel** (`line 35-38`):
  ```cpp
  q.parallel_for(range<1>(bufferCount), [=](id<1> idx) {
    dA[idx] = dtype(1.1);
    dB[idx] = dtype(1.1);
  })
  ```
  - 简单并行初始化，将所有元素设置为 1.1

- **Benchmark Kernel** (`line 41-53`):
  ```cpp
  q.parallel_for(nd_range<1>(range<1>(blockCount * blockSize), range<1>(blockSize)),
                 [=](nd_item<1> item) {
    int threadIdx = item.get_local_id(0);
    int blockIdx = item.get_group(0);

    dtype localSum = dtype(0);
    for (int i = 0; i < N / 2; i++) {
      int idx = (blockSize * blockRun * i + (blockIdx % blockRun) * blockSize) * 2 + threadIdx;
      localSum += dB[idx] * dB[idx + blockSize];
    }
    localSum *= (dtype)1.3;
    if (threadIdx > 1233 || localSum == (dtype)23.12)
      dA[threadIdx] += localSum;
  })
  ```
  - **Access Pattern**: 通过精心设计的索引计算 `idx`，控制内存访问的跨度
  - **Stride Calculation**:
    - `blockSize * blockRun * i`: 基础偏移，随 `i` 线性增长
    - `(blockIdx % blockRun) * blockSize`: 块内偏移，确保同一 work-group 内的线程访问连续内存
    - `* 2 + threadIdx`: 双倍跨度和线程内偏移
  - **Anti-Optimization**:
    - `localSum *= 1.3`: 防止编译器优化掉计算
    - `if (threadIdx > 1233 || localSum == 23.12)`: 永远为假的分支，防止死代码消除

## 3. API Interface

### 编译接口
```bash
# build.sh
clang++ -O3 \
  -fsycl \
  -fsycl-targets=nvptx64-nvidia-cuda \
  sycl-gpu-l2-cache.cpp \
  -o sycl-gpu-l2-cache \
  -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80

./sycl-gpu-l2-cache
```

**关键编译参数**:
- `-fsycl`: 启用 SYCL 语言支持
- `-fsycl-targets=nvptx64-nvidia-cuda`: 目标架构为 NVIDIA CUDA
- `--cuda-gpu-arch=sm_80`: 针对 Ampere 架构 (A100, RTX 30xx)
- `-O3`: 最高优化级别

### 运行时输出格式
```
Running on GPU: [GPU Name]

     data set   exec time     spread        Eff. bw
      [kB]      [kB]      [ms]         [%]      [GB/s]
    ...
```

**输出字段**:
- 第 1 列: 单次迭代的数据量 (kB) = `N * blockSize * sizeof(dtype)`
- 第 2 列: 总数据集大小 (kB) = 第 1 列 × `blockRun`
- 第 3 列: 最快执行时间 (ms)
- 第 4 列: 时间波动率 (%) = (max - min) / average_of_middle_9
- 第 5 列: 有效带宽 (GB/s) = 数据量 × blockCount / 时间

## 4. Usage Example

```bash
# 1. 编译程序
cd /home/qy/src/Infini/InfiniPerf/benchmarks/hardware/cuda/gpu-l2-cache/sycl
bash build.sh

# 2. 运行测试
./sycl-gpu-l2-cache

# 预期输出：
# Running on GPU: NVIDIA GeForce RTX 3090
#      data set   exec time     spread        Eff. bw
#        512 kB      1536 kB       45ms      2.3%      68.2 GB/s
#        512 kB      1688 kB       47ms      1.8%      72.1 GB/s
#        ...
```

## 5. Implementation Details

### 内存跨度控制算法
- **目的**: 通过逐渐增加 `blockRun` 参数，控制相邻 work-group 访问的内存地址间距
- **效果**:
  - 当 `blockRun` 较小时，所有 work-group 的访问都能命中 L2 缓存
  - 当 `blockRun` 增大到某个阈值，总数据集超过 L2 缓存容量，带宽骤降
  - 通过带宽突变点识别 L2 缓存大小

### 时间测量策略
- **多次采样**: 对每个数据集大小进行 11 次测试
- **取最小值**: 使用 `time[0]` (排序后的最小值) 作为有效执行时间，排除系统抖动
- **波动率计算**: 剔除最大最小值后计算剩余 9 次的相对波动，用于验证测试稳定性

### 数据类型选择
- `dtype = double`: 8 字节浮点数，提供较高的内存带宽压力
- 乘加操作 `dB[idx] * dB[idx + blockSize]` 确保实际的内存读写（不仅仅是加载）

### Work-Group 调度
- `nd_range<1>(blockCount * blockSize, blockSize)`:
  - 全局工作项: 200,000 × 1,024 = 204.8M
  - Work-group 大小: 1,024 (NVIDIA GPU 的典型 warp/wavefront 大小)
  - Work-group 数量: 200,000

### 编译器对抗技术
1. **分支混淆**: `if (threadIdx > 1233 || localSum == 23.12)` - 永假条件，防止死代码消除
2. **无用赋值**: `dA[threadIdx] += localSum` - 在永假分支中写入，防止优化掉 `localSum` 计算
3. **浮点常量**: `1.3` 和 `23.12` - 非整数值，防止常量折叠优化

### 性能计算公式
```cpp
// 数据量 (字节)
double blockDV = N * blockSize * sizeof(dtype);

// 有效带宽 (GB/s)
double bw = blockDV * blockCount / time[0] / 1.0e9;
```

**解释**:
- `blockDV`: 单次迭代每个 work-group 访问的数据量
- `blockDV * blockCount`: 所有 work-group 总访问量
- 除以最小时间得到峰值带宽

### L2 缓存识别原理
- **理论依据**: 当工作集 < L2 缓存时，带宽接近 L2 带宽；当工作集 > L2 缓存时，带宽降至显存带宽
- **实际应用**: 观察输出中带宽急剧下降的点，对应的 `blockDV * blockRun` 即为 L2 缓存容量的近似值
- **示例**: 如果带宽在 40 MB 处从 70 GB/s 降至 30 GB/s，则 L2 缓存约为 40 MB

### SYCL 特定实现
- **Queue 创建** (`line 20`):
  ```cpp
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  ```
  - `gpu_selector_v`: 强制选择 GPU 设备
  - `enable_profiling`: 启用性能分析属性（虽然本代码主要使用主机端计时）

- **设备内存管理**:
  - `malloc_device<T>(size, queue)`: 分配设备内存
  - `free(ptr, queue)`: 释放设备内存
  - 使用 SYCL 统一内存管理 API，无需显式 `cudaMemcpy`

- **Kernel 提交**:
  - `.wait()`: 同步等待 kernel 完成，确保计时准确性
  - `nd_range` vs `range`: 分别用于需要 work-group 信息和简单并行场景

### 依赖项
- **编译器**: clang++ (需支持 SYCL)
- **运行时**: oneAPI 或 DPC++ 运行时库
- **目标硬件**: NVIDIA GPU (compute capability 8.0+)
- **系统库**: 标准 C++ 库 (chrono, iomanip, algorithm, numeric)

### 设计模式
- **渐进式测试**: 指数增长的数据集大小，实现对数尺度的缓存扫描
- **统计采样**: 多次采样 + 中位数过滤，提高结果可靠性
- **零拷贝优化**: 所有计算在设备端完成，避免主机-设备数据传输开销
