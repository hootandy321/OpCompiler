# Causal Softmax Kunlun Implementation

昆仑卡(XPU)设备上的因果掩码softmax算子实现，专门针对Transformer推理中的注意力得分计算优化，实现了下三角掩码的softmax归一化操作。

## 1. 模块结构

- **`causal_softmax_kunlun.h`**: 昆仑算子描述符声明，使用宏定义接口
- **`causal_softmax_kunlun.xpu`**: 昆仑XPU设备的主实现文件，包含核函数启动逻辑和算子接口
- **`kernel.h`**: 核心计算内核实现，包含因果掩码softmax的设备端算法

## 2. 核心类

### `op::causal_softmax::kunlun::Descriptor`
- **位置**: `causal_softmax_kunlun.xpu`
- **主要功能**: 昆仑设备因果softmax算子的描述符类，继承自`InfiniopDescriptor`，负责算子创建、内存管理和内核调度
- **关键成员**:
  - `_opaque`: 指向`Opaque`结构体的指针，包含昆仑设备句柄的内部状态
  - `_info`: `CausalSoftmaxInfo`对象，存储张量形状、步长、数据类型等元信息
  - `_workspace_size`: 工作空间大小(当前实现为0)
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证输入输出张量描述符的兼容性(数据类型一致性、形状有效性)，创建算子描述符实例。验证规则：数据类型必须为F16/BF16/F32之一，张量维度必须为2或3，且最后一维必须大于等于倒数第二维
  - `calculate(workspace, workspace_size, y, x, stream_)`: 执行因果softmax计算的主入口，使用64核的BLOCK_SIZE启动内核，对每个batch中的序列独立计算
- **生命周期**: 通过create静态方法构造，析构函数释放内部opaque指针，采用RAII模式管理资源

### `causalSoftmaxKernel<BLOCK_SIZE, Tdata, Tcompute>`
- **位置**: `causal_softmax_kunlun.xpu`
- **主要功能**: XPU设备端全局内核函数，负责在全局内存和共享内存之间传输数据，并调度计算内核
- **关键参数**:
  - `BLOCK_SIZE`: 编译时常量，指定每个cluster的核心数(固定为64)
  - `Tdata`: 输入输出数据类型(half/bfloat16_t/float)
  - `Tcompute`: 计算类型(固定为float，用于精度保证)
- **核心逻辑**:
  1. 使用cluster_id()获取当前处理的行为标识
  2. 通过GM2SM_ASYNC异步从全局内存加载输入行到共享内存(x_sm)
  3. 调用causalSoftmaxBlock在共享内存中完成计算
  4. 通过SM2GM_ASYNC异步将结果从共享内存写回全局内存
  5. 使用sync_cluster()进行cluster内全局同步
- **内存层次**: 利用40KB的共享内存(SM_SIZE=40960)作为中间缓冲，减少全局内存访问

### `causalSoftmaxBlock<BLOCK_SIZE, Tdata, Tcompute>`
- **位置**: `kernel.h`
- **主要功能**: 共享内存中的因果掩码softmax计算内核，实现三阶段归约算法
- **算法流程**:

  **阶段1: 最大值归约(Reduce Max)**
  ```
  max_i = max(x_ij) for all valid j in row i
  ```
  - 使用`op::common_kunlun::reduce_op::max`进行跨核心归约
  - 有效元素数量: `width - height + 1 + row_id`(实现因果掩码)
  - 通过atomicMax保证多核同步更新共享最大值

  **阶段2: 指数变换与掩码(Exp & Mask)**
  ```
  for j in [0, width):
    if j + height <= width + row_id:  // 因果掩码条件
      y_ij = exp(x_ij - max_i)
    else:
      y_ij = 0  // 被掩码位置
  ```
  - 下三角掩码逻辑: 行i只能看到前`total_seq_len - seq_len + i + 1`个位置
  - 数据类型特化: half使用hexp(), bfloat16使用__bfloat162float转换, float直接exp()
  - 多核并行: 每个核心处理步长为BLOCK_SIZE的元素子集

  **阶段3: 求和归约与归一化(Reduce Sum & Normalize)**
  ```
  sum_i = sum(y_ij) for all j in row i
  y_ij = y_ij / sum_i  // 最终归一化
  ```
  - 使用`op::common_kunlun::reduce_op::sum`进行跨核心求和
  - 通过atomicAdd累加各核心的局部和
  - 处理sum=0的边界情况，避免除零错误
- **同步机制**: 在三个阶段之间使用sync_cluster()确保所有核心完成当前阶段后再进入下一阶段
- **时间复杂度**: O(width)每个元素处理常数时间，归约操作O(log BLOCK_SIZE)但BLOCK_SIZE固定为64

## 3. API接口

```cpp
// 创建算子描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // 昆仑设备句柄
    Descriptor **desc_ptr,                            // 输出: 创建的描述符指针
    infiniopTensorDescriptor_t y_desc,                // 输出张量描述符 [batch, seq_len, total_seq_len] 或 [seq_len, total_seq_len]
    infiniopTensorDescriptor_t x_desc                 // 输入张量描述符(形状与y相同)
);
// 返回: INFINI_STATUS_SUCCESS 或错误码(数据类型不匹配/形状无效)

// 执行因果softmax计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                                  // 工作空间(当前未使用, 传入nullptr)
    size_t workspace_size,                            // 工作空间大小(当前为0)
    void *y,                                          // 输出数据指针 [batch, seq_len, total_seq_len]
    const void *x,                                    // 输入数据指针(未归一化的logits)
    void *stream_                                     // 昆仑计算流
) const;
// 返回: INFINI_STATUS_SUCCESS 或 INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例: 在GPT-2推理中应用因果softmax到注意力得分
#include "infiniop/ops/causal_softmax/kunlun/causal_softmax_kunlun.h"

// 假设我们有一个batch_size=8, seq_len=128, total_seq_len=1024的attention scores
// 数据类型: half(FP16)以节省显存

// 1. 创建昆仑设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_KUNLUN, device_id);

// 2. 准备输入输出张量描述符
int64_t shape[3] = {8, 128, 1024};  // [batch, seq_len, total_seq_len]
int64_t stride[3] = {128*1024, 1024, 1};  // 连续内存布局

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc, INFINI_DTYPE_F16, 3, shape, stride);
infiniopCreateTensorDescriptor(handle, &y_desc, INFINI_DTYPE_F16, 3, shape, stride);

// 3. 创建因果softmax算子
op::causal_softmax::kunlun::Descriptor* softmax_op;
infiniStatus_t status = op::causal_softmax::kunlun::Descriptor::create(
    handle, &softmax_op, y_desc, x_desc
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误: 数据类型不支持或形状无效
}

// 4. 分配设备内存并初始化输入
half* d_x;
half* d_y;
xpuMalloc((void**)&d_x, 8 * 128 * 1024 * sizeof(half));
xpuMalloc((void**)&d_y, 8 * 128 * 1024 * sizeof(half));
// ... 将attention scores从主机复制到d_x ...

// 5. 获取昆仑流
kunlunStream_t stream;
xpuStreamCreate(&stream);

// 6. 执行因果softmax计算
status = softmax_op->calculate(
    nullptr,           // 无需工作空间
    0,                 // 工作空间大小为0
    d_y,               // 输出: 归一化后的attention weights
    d_x,               // 输入: 原始attention scores
    stream             // 昆仑计算流
);

// 7. 异步等待计算完成
xpuStreamSynchronize(stream);

// 8. 使用结果进行后续的矩阵乘法(attention_weights * value_vectors)
// ... 后续计算 ...

// 9. 清理资源
delete softmax_op;
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
xpuFree(d_x);
xpuFree(d_y);
xpuStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **共享内存策略**: 使用固定大小的共享内存缓冲区(40KB，由SM_SIZE定义)作为临时存储，每个cluster处理一行数据时复用同一块共享内存
- **数据传输优化**: 采用GM2SM_ASYNC/SM2GM_ASYNC异步数据传输，隐藏全局内存访问延迟
- **零拷贝工作空间**: 当前实现不需要额外工作空间，所有计算在共享内存中完成

### 并发机制
- **Cluster级并行**: 每行由一个独立的cluster处理，cluster_id()确定当前处理的行索引
- **核心级并行**: 每个cluster内64个核心并行处理行内的元素，使用core_id()进行元素分发
- **同步原语**:
  - `sync_cluster()`: cluster内全局栅栏，确保所有核心到达同步点
  - `atomicAdd/atomicMax`: 共享内存原子操作，使用ticket_lock_mix()实现基于票据锁的互斥访问
  - `mfence_sm()`: 共享内存写屏障，保证可见性
- **锁策略**: 对于half和bfloat16类型的原子操作，使用ticket_lock_mix/unlock_mix实现的自旋锁，避免硬件原生原子操作的精度损失

### 性能优化
- **SIMD向量化**:
  - half类型: 使用512位SIMD指令(32个half/向量)进行最大值归约，通过vload_lm_float16x32_mz/vvmax_float16x32_mz实现
  - float类型: 使用512位SIMD指令(16个float/向量)进行最大值归约，通过vload_lm_float32x16_mz/vvmax_float32x16_mz实现
- **类型提升策略**: 所有计算在float精度下执行(Tcompute=float)，避免半精度浮点的溢出和下溢问题，仅在存储时转换回原始数据类型
- **归约算法优化**:
  - 最大值归约: 先进行局部SIMD归约，再通过原子操作跨核心归约
  - 求和归约: 使用原子累加器聚合各核心局部和
- **分支优化**: 使用if constexpr编译期分支消除，避免数据类型判断的运行时开销

### 错误处理
- **类型检查**: create()方法验证输入输出数据类型一致性，仅支持INFINI_DTYPE_F16/F32/BF16
- **形状验证**:
  - 维度检查: 仅支持2D或3D张量
  - 因果约束: 要求shape[ndim-1] >= shape[ndim-2]，确保total_seq_len >= seq_len
- **边界条件**: calculate()方法中处理sum=0的情况，避免除零错误，返回全0张量

### 依赖项
- **昆仑XPU工具链**:
  - `xpu/runtime.h`: XPU运行时API
  - `xpu/kernel/xtdk.h`: XPU内核开发工具包
  - `xpu/kernel/xtdk_bf16.h`: BFloat16数学函数支持
  - `xpu/kernel/xtdk_math.h`: 通用数学函数(exp, fmax等)
  - `xpu/kernel/xtdk_simd.h`: SIMD向量指令集
- **内部依赖**:
  - `device::kunlun::Handle`: 昆仑设备句柄管理
  - `device::kunlun::kernel`: 昆仑内核公共函数(原子操作、类型转换、内存操作)
  - `op::common_kunlun::reduce_op`: 跨核心归约操作(max/sum)

### 设计模式
- **策略模式**: 通过模板参数BLOCK_SIZE/Tdata/Tcompute实现编译期多态，同一套代码支持不同数据类型和块大小
- **CRTP(奇异递归模板模式)**: DESCRIPTOR宏通过命名空间注入生成特定设备的Descriptor类
- **RAII**: Descriptor类管理opaque指针生命周期，析构函数自动释放资源
- **工厂模式**: create()静态方法作为构造工厂，集中处理验证和初始化逻辑
- **模板特化**: 为half和float类型特化max()函数，利用SIMD指令优化性能

### 算法特点
- **因果掩码实现**: 通过条件`width + row_id >= col + height`实现下三角掩码，比传统mask张量乘法节省内存访问
- **数值稳定性**: 采用经典softmax稳定算法(max(x) - x)，避免exp()溢出
- **批处理策略**: 外层循环遍历batch维度，每个序列独立启动一个内核，便于流式处理
