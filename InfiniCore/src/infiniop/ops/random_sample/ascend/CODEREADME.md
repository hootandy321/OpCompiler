# Ascend Random Sample 核心实现文档

该模块实现了华为昇腾(Ascend) NPU 上的随机采样算子，结合 ACLNN 高级 API 和 AscendC 自定义内核，为 LLM 推理中的 Top-K/Top-P (nucleus) 采样提供硬件加速支持。

## 1. 模块结构

- **`random_sample_aclnn.h`**: 昇腾后端的描述符声明文件，通过宏定义生成 Descriptor 类
- **`randomsample_aclnn.cc`**: 主实现文件，封装 ACLNN TopK 算子和自定义采样内核的调度逻辑
- **`random_sample_kernel.cpp`**: AscendC 自定义内核实现，包含 SoftMax、前缀和采样等核心算法

## 2. 核心类

### `Descriptor`
- **位置**: `randomsample_aclnn.cc`
- **主要功能**: 管理 Ascend 平台上的随机采样算子生命周期，协调 ACLNN TopK 和自定义内核的执行
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构体的指针，封装 ACLNN 张量描述符
  - `_info`: `RandomSampleInfo` 结构，存储输入/输出张量的数据类型和维度信息
  - `_min_workspace_size`: 最小工作空间大小，用于存储 TopK 结果和临时数据
- **核心方法**:
  - `create(handle, desc_ptr, result_desc, probs_desc)`: 构造函数，验证数据类型（输出必须为 INT64，输入支持 FP16/FP32），计算工作空间大小（`probs_numel * (sizeof(dt_p) + sizeof(int64_t))`），初始化 ACLNN 张量描述符
  - `minWorkspaceSize()`: 返回工作空间大小，O(1) 时间复杂度
  - `calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream)`: 执行采样计算，根据条件决定是否调用自定义采样内核
- **生命周期**: 通过 `create()` 静态工厂方法构造，析构时自动释放 `_opaque` 中持有的 ACLNN 描述符

### `Descriptor::Opaque`
- **位置**: `randomsample_aclnn.cc`
- **主要功能**: 封装 Ascend ACLNN API 的张量描述符，隐藏底层 ACL 细节
- **关键成员**:
  - `probs`: `aclnnTensorDescriptor_t`，输入概率分布的 ACL 张量描述符
  - `result`: `aclnnTensorDescriptor_t`，输出采样结果的 ACL 张量描述符
- **析构行为**: 自动释放 `probs` 和 `result` 指向的 `aclnnTensorDescriptor` 对象

### `RandomSampleKernel<T>`
- **位置**: `random_sample_kernel.cpp`
- **主要功能**: AscendC 向量编程模板类，实现核内(random_sample_kernel_fp16/fp32)的完整采样流程
- **关键成员**:
  - `_pGM`, `_topk_valGM`, `_topk_idxGM`, `_resGM`: 全局内存张量，分别指向原始概率、TopK 值、TopK 索引、采样结果
  - `pipe`: AscendC 流水线对象，管理输入/输出队列
  - `_topk_valQue`, `_topk_idxQue`, `_resQue`: 双缓冲队列，实现数据流水线传输
  - `_inBuf`, `_tmp1Buf`, `_tmp2Buf`, `_tmp3Buf`, `_softmax_OutBuf`, `_inclusive_sum_OutBuf`: 本地缓冲区，用于核内计算
  - `_topk`, `_voc`: TopK 个数和词汇表大小
  - `_random_val`, `_topp`, `_invTemp`: 采样参数（随机值、Top-P 阈值、温度倒数）
  - `_negMax`, `_sum`: 数值稳定化参数（负最大值、指数和）
- **核心方法**:
  - `init(probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk, temperature, n)`: 初始化内核，对齐数据到 32 字节边界（`alignTileLen`），分配队列和缓冲区
  - `process()`: 主执行流程，依次调用 `copyIn()`, `compute()`, `copyOut()`
  - `SoftMax(topkValIn, softMaxOut)`: 数值稳定的 SoftMax 实现，先减最大值，再按温度缩放，最后指数化和归一化
  - `InclusiveSum(topkValIn, topkValOut)`: 使用 AscendC `CumSum` API 计算前缀和，配置为包含性前缀和（`CumSumConfig{true, false, false}`）
  - `RandomSample(valIn, Index, result)`: Top-P 采样核心算法，在前缀和数组中二分查找确定采样位置
  - `copyIn()`: 从全局内存拷贝 TopK 结果到本地队列，同时遍历全量概率计算指数和 `_sum`
  - `compute()`: 依次执行 SoftMax、前缀和、随机采样三个阶段
  - `copyOut()`: 将采样结果从本地队列写回全局内存
- **模板特化**: 支持 `half` (FP16) 和 `float` (FP32) 两种数据类型

## 3. API 接口

```cpp
// 创建算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,              // Ascend 设备句柄
    Descriptor **desc_ptr,                // 输出：描述符指针
    infiniopTensorDescriptor_t result_desc, // 输出张量描述符（必须是 INT64 标量）
    infiniopTensorDescriptor_t probs_desc  // 输入概率张量描述符（1D，支持 FP16/FP32）
);
// 返回 INFINI_STATUS_SUCCESS 或错误码

// 查询工作空间大小
size_t Descriptor::minWorkspaceSize() const;
// 返回：probs_numel * (sizeof(dt_p) + sizeof(int64_t))

// 执行随机采样计算
infiniStatus_t Descriptor::calculate(
    void *workspace,        // 工作空间指针，至少 minWorkspaceSize() 字节
    size_t workspace_size,  // 工作空间大小
    void *result,           // 输出：采样结果（int64 标量）
    const void *probs,      // 输入：概率分布（1D 数组）
    float random_val,       // 随机数 [0, 1)
    float topp,             // Top-P 阈值
    int topk,               // Top-K 个数
    float temperature,      // 温度参数
    void *stream            // Ascend 流
) const;
// 返回 INFINI_STATUS_SUCCESS 或错误码

// AscendC 自定义内核入口（由 random_sample_kernel_launch 调用）
extern "C" __global__ __aicore__ void random_sample_kernel_fp16(
    GM_ADDR probs,          // 输入：概率分布（FP16）
    GM_ADDR result,         // 输出：采样结果
    GM_ADDR topk_val_addr,  // 输入：TopK 值
    GM_ADDR topk_idx_addr,  // 输入：TopK 索引
    float random_val,       // 随机数
    float topp,             // Top-P 阈值
    int topk,               // Top-K 个数
    float temperature,      // 温度
    int32_t n               // 词汇表大小
);

extern "C" __global__ __aicore__ void random_sample_kernel_fp32(
    // 参数同上，但 probs 为 FP32
);
```

## 4. 使用示例

```cpp
// 示例：在 Ascend NPU 上执行 Top-K/Top-P 采样
#include "random_sample_aclnn.h"
#include <acl/acl.h>

// 1. 准备输入张量描述符
infiniopTensorDescriptor_t probs_desc, result_desc;
// probs: shape [vocab_size], dtype FP16, stride [1]
// result: shape [], dtype INT64
// (初始化代码省略)

// 2. 创建算子描述符
op::random_sample::ascend::Descriptor *desc;
infiniStatus_t status = op::random_sample::ascend::Descriptor::create(
    ascend_handle, &desc, result_desc, probs_desc
);

// 3. 分配工作空间
size_t workspace_size = desc->minWorkspaceSize();
void *workspace;
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);

// 4. 准备设备内存
void *d_probs, *d_result;
aclrtMalloc(&d_probs, probs_size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_result, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(d_probs, probs_size, h_probs, probs_size, ACL_MEMCPY_HOST_TO_DEVICE);

// 5. 执行采样（temperature=1.0, topk=50, topp=0.9）
float random_val = 0.1234f; // 从外部随机数生成器获取
status = desc->calculate(
    workspace, workspace_size,
    d_result, d_probs,
    random_val, 0.9f, 50, 1.0f,
    stream
);

// 6. 同步并取回结果
aclrtSynchronizeStream(stream);
int64_t token_id;
aclrtMemcpy(&token_id, sizeof(int64_t), d_result, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

// 7. 清理资源
delete desc;
aclrtFree(workspace);
aclrtFree(d_probs);
aclrtFree(d_result);
```

## 5. 实现细节

- **内存管理策略**:
  - 使用两级内存分配：主工作空间由调用方提供，内核内部临时工作空间通过 `aclrtMalloc` 动态分配并立即释放
  - TopK 临时存储布局：前 `topk * sizeof(dt_p)` 字节存储 TopK 值，后 `topk * sizeof(int64_t)` 字节存储 TopK 索引
  - AscendC 内核使用双缓冲队列（TQue）和本地缓冲区（TBuf）隐藏内存传输延迟

- **并发控制**:
  - 通过 Ascend 流（stream）实现执行顺序保证
  - 调用 `aclSetAclOpExecutorRepeatable` 确保 TopK 算子可重复执行（多线程安全）
  - ACLNN API 和 AscendC 内核在同一个流中串行执行，通过流同步保证数据依赖

- **性能优化技术**:
  - **快速路径**: 当 `random_val=0 || topp=0 || topk=1 || temperature=0` 时，直接使用 TopK 的 argmax 结果，跳过 SoftMax 和采样
  - **内存对齐**: 所有本地缓冲区按 32 字节（`BYTE_ALIGN`）对齐，利用 Ascend AI Core 的向量加载单元
  - **流水线并行**: `copyIn` 和 `copyOut` 使用双缓冲队列，计算与数据传输重叠
  - **块处理**: 在 `copyIn` 中以 256 个元素（`BLOCK_LEN`）为单位遍历全量概率，减少循环开销
  - **数据类型**: 核内计算使用模板特化，FP16/FP32 各有独立内核实例，避免类型分支

- **数值稳定性**:
  - SoftMax 实现采用经典稳定化技巧：先减去最大值（`_negMax`），再按温度缩放和指数化
  - 前缀和使用 AscendC `CumSum` API，保证累积误差最小

- **错误处理**:
  - `create()` 阶段检查数据类型约束（输出必须是 INT64，输入支持 FP16/FP32），返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - `calculate()` 验证工作空间大小，不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
  - ACLNN API 调用使用 `CHECK_ACL` 宏封装，自动打印错误消息（`GetRecentErrMsg()`）
  - 自定义内核启动时检查数据类型，不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

- **依赖关系**:
  - **外部依赖**: Ascend ACLNN (`aclnnTopk`, `aclnnTopkGetWorkspaceSize`), AscendC (`__aicore__`, `DataCopy`, `Adds`, `Muls`, `Exp`, `CumSum`), ACL Runtime (`aclrtMalloc`, `aclrtFree`)
  - **内部依赖**: `random_sample.h`（描述符基类宏定义）, `info.h`（RandomSampleInfo 结构）, `common_ascend.h`（ACLNN 张量描述符封装）, `ascend_kernel_common.h`（AscendC 通用常量和工具函数）

- **设计模式**:
  - **策略模式**: `calculate()` 根据 `dosample` 标志选择快速路径（直接 TopK）或完整采样路径
  - **模板方法模式**: `RandomSampleKernel::process()` 定义算法骨架（copyIn -> compute -> copyOut），子步骤由各方法实现
  - **RAII**: `Descriptor::Opaque` 的析构函数自动释放 ACLNN 描述符，避免资源泄漏
  - **工厂方法**: `Descriptor::create()` 静态工厂封装复杂的对象构造逻辑

- **算法复杂度**:
  - **TopK 阶段**: 由 ACLNN 实现，复杂度 O(vocab_size log topk)
  - **SoftMax 阶段**: O(topk)
  - **前缀和阶段**: O(topk)（使用 AscendC 并行前缀和）
  - **采样阶段**: O(topk)（线性扫描查找）
  - **全量概率遍历**: O(vocab_size)，以 256 元素为块向量化处理
  - **总体**: O(vocab_size log topk + vocab_size) = O(vocab_size log topk)

- **硬件适配**:
  - 针对 Ascend AI Core 的向量处理单元优化，使用 32 字节对齐和 256 元素块大小
  - 利用 AscendC 的双缓冲机制隐藏 GM（Global Memory）到本地内存的传输延迟
  - 核内并行通过 AscendC 的向量化 API（`Adds`, `Muls`, `Exp`）自动实现
