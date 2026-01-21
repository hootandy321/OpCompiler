# AWQ 权重量化解算子 NVIDIA GPU 实现文档

## 模块概述

本模块实现了 AWQ (Activation-aware Weight Quantization) 算法的权重解量化 CUDA 内核，专门用于将 4-bit 量化权重恢复为 FP16 格式。该实现针对 NVIDIA GPU 进行了深度优化，利用 PTX 内联汇编和向量化指令实现高吞吐量的权重量化/解量化操作，是 LLM 推理中关键的预处理算子。

核心功能：将 int4 量化权重（packed 格式）减去 zero-point 并乘以缩放因子，输出 FP16 精度的权重矩阵。

## 1. 文件结构

- **`dequantize_w42f16_nvidia.cuh`**：NVIDIA 后端的描述符声明，通过 `DESCRIPTOR(nvidia)` 宏展开生成完整的 Descriptor 类接口
- **`dequantize_w42f16_kernel.cuh`**：核心解量化内核函数实现，包含设备端 `dequantize_s4_to_fp16x2()` 函数，负责 4-bit 到 FP16x2 的高效转换
- **`dequantize_w42f16_nvidia.cu`**：主 CUDA 实现文件，包含内核启动逻辑、描述符方法实现（create/calculate）以及针对不同 GPU 架构的双路优化代码

## 2. 核心类与函数

### 设备端函数：`dequantize_s4_to_fp16x2()`
- **位置**：`dequantize_w42f16_kernel.cuh:3-125`
- **功能**：将 1 个 32-bit 无符号整数（包含 8 个 packed 4-bit 整数）转换为 1 个 `uint4`（包含 8 个 FP16 半精度浮点数）
- **输入**：`uint32_t const &source` - 内存布局为 `[v7, v6, v5, v4, v3, v2, v1, v0]`，每个 v 占 4-bit
- **输出**：`uint4` - 8 个 FP16 值的向量化容器
- **关键实现细节**：
  - **双路径优化**：针对不同 GPU 架构提供两套实现
    - **计算能力 < 7.5 (旧架构)**：使用标准 CUDA API (`__half`, `__halves2half2`)
      - 手动位提取：通过移位和掩码获取 8 个 4-bit 无符号整数
      - 符号转换：减去偏移量 8 映射到有符号范围 [-8, 7]
      - 数据重排：按 PTX 交织顺序打包为 `__half2` 格式
    - **计算能力 >= 7.5 (Turing 及更新架构)**：使用 PTX 内联汇编进行极致优化
      - **LUT 三操作数逻辑**：通过 `lop3.b32` 指令并行提取 4-bit 元素，仅需 1 次移位指令
      - **魔法常数**：
        - `I4s_TO_F16s_MAGIC_NUM = 0x64006400` (FP16 格式的 100.0)
        - `FP16_TOP_MAGIC_NUM = 0x64006400` (FP16 格式的 1024.0，Haotian 修改为 1024)
        - `ONE_SIXTEENTH = 0x2c002c00` (FP16 格式的 1/16)
        - `NEG_64 = 0xd400d400` (FP16 格式的 -64.0)
      - **FMA 优化**：对 elt_23 和 elt_67 使用 `fma.rn.f16x2` 指令融合乘加操作，避免额外移位
  - **输出布局**：`result_ptr[0..3]` 按 `[low=vi, high=v{i+4}]` 交织存储

### CUDA 内核：`dequantize_weights()`
- **位置**：`dequantize_w42f16_nvidia.cu:12-124`
- **功能**：执行 AWQ 解量化主内核，处理 int4 权重 -> FP16 权重的完整流程
- **内核签名**：
  ```cuda
  __global__ void __launch_bounds__(64)
  dequantize_weights(
      int *__restrict__ B,              // 输入：packed int4 权重
      half *__restrict__ scaling_factors, // 输入：缩放因子 (FP16)
      int *__restrict__ zeros,           // 输入：zero point (packed int4)
      half *__restrict__ C,              // 输出：解量化后的 FP16 权重
      int group_size)                    // 分组大小
  ```
- **线程配置**：`__launch_bounds__(64)` - 每个块最多 64 个线程，编译器优化提示
- **分块策略**：
  - **固定块大小**：`BLOCK_X = BLOCK_Y = 8`（编译时常量）
  - **网格维度**：
    - X 方向块数 = `(out_features + 7) / 8`
    - Y 方向块数 = `(in_features + 7) / 8`
  - **每个线程处理 8 个元素**，实现 8x8 线程块的高效内存访问模式
- **内存访问模式**：
  - **索引计算**：
    - 输出索引：`index1 = 8*col + 8*row*N`（每个线程写 8 个 FP16）
    - 权重索引：`index2 = col + row*N`（每个线程读 1 个 packed int32）
    - zero point 索引：`index3 = col + (row/group_size)*N`（按组共享）
    - 缩放因子索引：`index4 = 8*col + (row/group_size)*N*8`（每个组 8 个 scale）
  - **数据加载**：
    - 使用 `uint32_t` 指针加载 packed 权重和 zero point（每个 uint32 包含 8 个 int4）
    - 使用 `uint4` 指针加载 8 个 FP16 缩放因子（128-bit 加载）
  - **共享内存**：声明 `half B_shared[32 * (128 + 8)]`（实际仅使用前 8 个元素）
- **解量化流程**（以计算能力 >= 7.5 为例）：
  1. **解包 int4**：调用 `dequantize_s4_to_fp16x2()` 将 packed int32 转换为 8 个 FP16
  2. **减去 zero point**：使用 PTX `sub.f16x2` 指令向量减法
  3. **乘以缩放因子**：使用 PTX `fma.rn.f16x2` 指令融合乘加（multiply-add with zero）
  4. **写回结果**：先将结果写入共享内存，再循环写入全局内存（8 次 FP16 写入）

### 主机端类：`op::dequantize_awq::nvidia::Descriptor`
- **位置**：`dequantize_w42f16_nvidia.cu:127-192`
- **父类**：`InfiniopDescriptor`（通用算子描述符基类）
- **不透明成员**：
  ```cpp
  struct Opaque {
      std::shared_ptr<device::nvidia::Handle::Internal> internal;
  };
  Opaque *_opaque;
  ```
  - 持有 NVIDIA 设备句柄的内部实现（智能指针管理）
- **数据成员**：
  - `DequantizeAWQInfo _info`：张量形状、分组信息等元数据
  - `size_t _workspace_size`：工作空间大小（当前实现为 0）
- **核心方法**：
  - **`create()`** (静态工厂方法)：
    - **功能**：创建描述符实例，验证张量形状兼容性
    - **参数**：
      - `infiniopHandle_t handle_`：设备句柄（NVIDIA Handle）
      - `Descriptor **desc_ptr`：输出描述符指针
      - 4 个张量描述符（输出、量化权重、缩放因子、zero point）
    - **实现逻辑**：
      1. 转换句柄类型为 `device::nvidia::Handle*`
      2. 调用 `DequantizeAWQInfo::create()` 验证并提取形状信息
      3. 构造 Descriptor 实例（工作空间大小=0）
      4. 返回 `INFINI_STATUS_SUCCESS`
    - **复杂度**：O(1)
  - **`calculate()`**：
    - **功能**：启动 CUDA 内核执行解量化
    - **参数**：
      - `void *workspace, size_t workspace_size`：工作空间（未使用）
      - `void *out`：输出 FP16 权重矩阵
      - `const void *qweight, scales, zeros`：输入张量
      - `void *stream`：CUDA 流
    - **实现逻辑**：
      1. 从 `_info` 提取特征维度和分组信息
      2. 计算网格/块维度（固定 8x8 分块）
      3. 转换指针类型（`void*` -> `half*`/`int*`）
      4. 启动 `dequantize_weights<<<>>>` 内核
      5. 返回 `INFINI_STATUS_SUCCESS`
    - **性能特征**：
      - **计算密度**：每个线程处理 8 个元素，隐藏内存延迟
      - **内存带宽**：主要瓶颈为全局内存访问（读取 packed int4、scale、zeros，写入 FP16）
      - **占用率**：8x8 线程块 = 64 线程/块，适合现代 GPU
    - **错误处理**：当前实现不检查 CUDA 错误（依赖外部错误检查）

## 3. API 接口

```cpp
// 描述符创建接口
infiniStatus_t op::dequantize_awq::nvidia::Descriptor::create(
    infiniopHandle_t handle,              // NVIDIA 设备句柄
    Descriptor **desc_ptr,                // 输出：描述符指针
    infiniopTensorDescriptor_t out_desc,  // 输出张量：[out_features, in_features] FP16
    infiniopTensorDescriptor_t qweight_desc, // 输入：量化权重 [out_features/8, in_features] int32
    infiniopTensorDescriptor_t scales_desc,  // 输入：缩放因子 [out_features/8, in_features/group_size] FP16
    infiniopTensorDescriptor_t zeros_desc    // 输入：zero point [out_features/8, in_features/group_size] int32
);
// 返回：INFINI_STATUS_SUCCESS（或张量形状不匹配时的错误码）

// 解量化计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,           // 未使用，传 nullptr
    size_t workspace_size,     // 未使用，传 0
    void *out,                 // 输出：FP16 权重矩阵，形状 [out_features, in_features]
    const void *qweight,       // 输入：packed int4 权重，每个 int32 包含 8 个 int4
    const void *scales,        // 输入：FP16 缩放因子，按组广播
    const void *zeros,         // 输入：packed int4 zero point
    void *stream              // CUDA 流（cudaStream_t）
) const;
// 返回：INFINI_STATUS_SUCCESS
```

## 4. 使用示例

```cpp
// 示例：在 LLM 推理中解量化 AWQ 4-bit 权重
#include "infiniop/ops/dequantize_awq/nvidia/dequantize_w42f16_nvidia.cuh"

// 1. 创建张量描述符
constexpr int out_features = 4096;  // 隐藏层维度
constexpr int in_features = 4096;   // 输入维度
constexpr int group_size = 128;     // AWQ 分组大小

infiniopTensorDescriptor_t out_desc, qweight_desc, scales_desc, zeros_desc;
infiniopCreateTensorDescriptor(&out_desc, kInfiniDeviceCUDA, kInfiniFP16,
                              {out_features, in_features}, 2);  // [4096, 4096]
infiniopCreateTensorDescriptor(&qweight_desc, kInfiniDeviceCUDA, kInfiniInt32,
                              {out_features / 8, in_features}, 2);  // [512, 4096] packed
infiniopCreateTensorDescriptor(&scales_desc, kInfiniDeviceCUDA, kInfiniFP16,
                              {out_features / 8, in_features / group_size}, 2);  // [512, 32]
infiniopCreateTensorDescriptor(&zeros_desc, kInfiniDeviceCUDA, kInfiniInt32,
                              {out_features / 8, in_features / group_size}, 2);  // [512, 32]

// 2. 创建描述符
op::dequantize_awq::nvidia::Descriptor *dequant_desc;
infiniStatus_t status = op::dequantize_awq::nvidia::Descriptor::create(
    handle, &dequant_desc, out_desc, qweight_desc, scales_desc, zeros_desc);

// 3. 分配 GPU 内存
half *d_out;
int *d_qweight, *d_zeros;
half *d_scales;
cudaMalloc(&d_out, out_features * in_features * sizeof(half));
cudaMalloc(&d_qweight, (out_features / 8) * in_features * sizeof(int));
cudaMalloc(&d_scales, (out_features / 8) * (in_features / group_size) * sizeof(half));
cudaMalloc(&d_zeros, (out_features / 8) * (in_features / group_size) * sizeof(int));

// 4. 拷贝量化数据到 GPU（假设 h_qweight, h_scales, h_zeros 已准备）
cudaMemcpy(d_qweight, h_qweight, qweight_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_scales, h_scales, scales_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_zeros, h_zeros, zeros_size, cudaMemcpyHostToDevice);

// 5. 执行解量化
cudaStream_t stream;
cudaStreamCreate(&stream);
status = dequant_desc->calculate(nullptr, 0, d_out, d_qweight, d_scales, d_zeros, stream);

// 6. 同步并使用解量化后的权重
cudaStreamSynchronize(stream);
// 现在 d_out 可用于矩阵乘法内核

// 7. 清理
cudaFree(d_out);
cudaFree(d_qweight);
cudaFree(d_scales);
cudaFree(d_zeros);
delete dequant_desc;
```

## 5. 实现细节

### 算法优化
- **向量化内存访问**：
  - 使用 `uint4` (128-bit) 加载缩放因子，一次读取 8 个 FP16 值
  - 使用 `uint32_t` 加载 packed int4 权重，充分利用内存带宽
  - 输出时使用 `uint4` 存储指令，减少全局内存写入次数
- **指令级并行**：
  - PTX `lop3.b32` 指令实现 3 输入逻辑运算，单指令完成掩码+或运算
  - FMA 指令融合乘加操作，减少指令数量和寄存器压力
  - `sub.f16x2` 和 `fma.rn.f16x2` 实现 SIMD 并行计算（每指令处理 2 个 FP16）
- **寄存器优化**：
  - 避免不必要的移位指令（通过魔法常数和 FMA 技巧）
  - 使用常量表达式（`static constexpr`）在编译期计算掩码和魔法常数
- **分组量化**：
  - 支持可配置的 `group_size`（通常 128），每 `in_features/group_size` 行共享一套 scale/zero
  - 通过整数除法 `row / group_size` 计算组索引

### 内存管理
- **工作空间**：当前实现不使用工作空间（`workspace_size = 0`）
- **共享内存**：每个线程块声明 `32 * (128 + 8) = 4352` 字节共享内存，但仅使用前 128 字节（8 个 half）
- **内存对齐**：所有全局内存访问均自然对齐（uint32 和 uint4 要求 4/16 字节对齐）
- **数据打包格式**：
  - **int4 权重**：每个 int32 包含 8 个 int4，字节序为小端，最低字节包含 v0-v3
  - **FP16 缩放因子**：连续存储，每 8 个值为一组（对应 8 列输出）
  - **zero point**：与 int4 权重相同的打包格式

### 并发与线程安全
- **CUDA 内核并发**：每个线程块独立处理 8x8 输出瓦片，无需跨块同步
- **线程安全**：
  - 描述符对象不可变（`_info` 在构造后不变）
  - `calculate()` 方法为 const，无共享状态修改
  - 不同 CUDA 流可以并发执行不同计算任务
- **原子操作**：不使用原子操作（无跨线程数据竞争）

### 性能特征
- **时间复杂度**：O(out_features * in_features) - 每个元素处理常数时间
- **空间复杂度**：O(1) 额外空间（仅寄存器和共享内存）
- **内存带宽**：
  - 读取：每个输出元素需读取 1/8 个 int32（qweight）+ 1 个 FP16（scale）+ 1/8 个 int32（zero）
  - 写入：每个输出元素写入 1 个 FP16
  - **计算强度**：~8 FLOPs/元素（减法+乘法）+ 数据转换，属于计算密集型
- **占用率优化**：
  - 每个线程块 64 线程，在 SM 上可调度多个块
  - 寄存器使用：每个线程约需 10-15 个寄存器（估计）
  - 共享内存：每个块 4.5KB，允许每 SM 调度多个块（现代 GPU 共享内存 48-164KB）

### 错误处理
- **张量形状验证**：在 `create()` 中通过 `DequantizeAWQInfo::create()` 验证
  - 检查维度数、数据类型、分组大小匹配
- **CUDA 错误传播**：当前实现不检查 CUDA 内核启动错误
  - 假设调用者通过 `cudaGetLastError()` 或 `cudaStreamSynchronize()` 检查错误
- **边界条件**：
  - 支持任意特征维度（通过网格自动计算）
  - 要求 `in_features` 能被 `group_size` 整除
  - 要求 `out_features` 能被 8 整除（packed int4 格式要求）

### 架构适配性
- **多架构支持**：
  - **计算能力 < 7.5**：使用标准 CUDA API，兼容 Volta (V100) 及更早架构
  - **计算能力 >= 7.5**：使用 PTX 内联汇编，优化 Turing (RTX 20 系列)、Ampere (A100)、Hopper (H100)
- **指令集特性**：
  - 利用 `fma.rn.f16x2`（Turing+）进行半精度融合乘加
  - 使用 `lop3.b32` 进行三输入逻辑运算（所有架构）
- **性能端口性**：
  - 旧架构实现功能正确但性能较低
  - 新架构实现接近理论峰值内存带宽

### 依赖关系
- **外部依赖**：
  - `cuda_fp16.h`：FP16 数据类型和数学函数
  - `device/nvidia/nvidia_handle.cuh`：NVIDIA 设备句柄
  - `device/nvidia/nvidia_kernel_common.cuh`：通用内核工具
- **内部依赖**：
  - `../dequantize_awq.h`：算子接口定义宏
  - `../info.h`：`DequantizeAWQInfo` 类（形状验证）
  - `../../tensor.h`：张量描述符定义
  - `../../operator.h`：算子基类
- **编译选项**：
  - 需定义 `ENABLE_NVIDIA_API` 或 `ENABLE_QY_API` 才编译此模块

### 设计模式
- **策略模式**：通过计算能力宏实现多架构策略（旧架构 vs 新架构）
- **工厂模式**：`create()` 静态方法作为描述符工厂
- **不透明指针模式**：`Opaque` 结构体隐藏 NVIDIA 句柄实现细节
- **RAII**：`Descriptor::~Descriptor()` 自动释放 `_opaque` 内存
- **编译期多态**：`DESCRIPTOR(NAMESPACE)` 宏为不同后端生成相同接口类

### 魔法常数说明
- **`0x64006400`**：FP16 格式的 1024.0（两个 1024.0 打包为 `half2`）
  - 二进制：`0110 0100 0000 0000` (FP16)
  - 用于将无符号整数 [0, 15] 转换为有符号整数 [-8, 7]
- **`0x2c002c00`**：FP16 格式的 1/16 (0.0625)
  - 用于 FMA 指令实现除法（除以 16 = 乘以 1/16）
- **`0xd400d400`**：FP16 格式的 -64.0
  - 用于 FMA 指令：`(x * (1/16)) + (-64)` 实现 `x/16 - 64`
- **`0xf0 & 0xcc | 0xaa`**：LUT 操作数
  - 实现逻辑函数 `(A & B) | (~A & C)`（选择操作）
  - 用于提取特定 4-bit 字段

### 性能调优参数
- **块大小选择**：8x8 = 64 线程/块
  - 较小块大小提高占用率，但增加内核启动开销
  - 较大块大小减少启动开销，但降低占用率
  - 8x8 是经验最优值（平衡 warp 调度和内存合并）
- **每个线程处理元素数**：8 个元素
  - 提高计算密度，隐藏内存延迟
  - 与 uint4 向量化宽度匹配（128-bit = 8 x FP16）
- **分组大小**：通常 128（可配置）
  - 较小分组提高量化精度但增加 scale/zero 内存占用
  - 较大分组减少内存占用但降低精度
  - 128 是 AWQ 论文推荐值（平衡精度和效率）
