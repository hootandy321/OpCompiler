# RoPE CPU 后端核心实现文档

本模块实现了旋转位置编码（Rotary Position Embedding, RoPE）的 CPU 后端，支持 GPT-J 和 GPT-NeoX 两种主流算法风格，提供高性能的多线程并行计算。

## 1. 模块结构

- **`rope_cpu.h`**: CPU 后端声明文件，通过宏定义 `DESCRIPTOR(cpu)` 展开完整的 Descriptor 类声明
- **`rope_cpu.cc`**: CPU 后端核心实现，包含算子创建、RoPE 计算核心算法和类型分发逻辑

## 2. 核心类

### `op::rope::cpu::Descriptor`
- **位置**: `rope_cpu.cc` (通过 `rope_cpu.h` 宏展开定义)
- **主要功能**: RoPE 算子的 CPU 后端描述符，管理算子元数据和执行计算
- **继承体系**: 继承自 `InfiniopDescriptor`，包含 `RoPEInfo` 元数据结构
- **关键成员**:
  - `_opaque`: 不透明指针（本实现中未使用，设为 `nullptr`）
  - `_info`: `RoPEInfo` 结构体，存储张量形状、步长、数据类型等元数据
  - `_workspace_size`: 工作空间大小（本实现固定为 0）

### `RoPEInfo`
- **位置**: 定义在父目录 `rope.h` 中
- **主要功能**: 封装 RoPE 算子的所有静态元数据，验证张量形状兼容性
- **核心成员**:
  - `data_type`: 数据类型（支持 fp16、bf16、fp32、fp64）
  - `pos_type`: 位置 ID 类型（支持所有整数类型：int8/16/32/64、uint8/16/32/64）
  - `batch`: batch 大小（3D 张量时为 1，4D 张量时为实际值）
  - `seqlen`: 序列长度
  - `nhead`: 注意力头数
  - `dhead`: 每个头的维度（必须是 `table_dim * 2`）
  - `table_len`: sin/cos 表的长度（最大位置编码数）
  - `table_dim`: sin/cos 表的维度（等于 `dhead / 2`）
  - `x_stride_batch/x_stride_seqlen/x_stride_nhead`: 输入张量各维步长
  - `y_stride_batch/y_stride_seqlen/y_stride_nhead`: 输出张量各维步长
  - `has_batch_dim`: 是否包含 batch 维度（4D 为 true，3D 为 false）
  - `pos_has_batch_dim`: 位置 ID 是否为 2D（per-batch）或 1D（共享）
  - `algo`: RoPE 算法类型（GPT-J 或 GPT-NeoX）

- **生命周期**:
  - **构造**: 通过 `RoPEInfo::createRoPEInfo()` 静态工厂方法创建，执行严格的形状验证
  - **验证规则**:
    - 输入/输出张量必须同时为 3D（无 batch）或 4D（有 batch）
    - 位置 ID 可为 1D（所有 batch 共享）或 2D（每个 batch 独立）
    - sin/cos 表必须是 2D 且完全连续存储
    - `dhead` 必须等于 `table_dim * 2`
    - 张量最后一维必须连续（stride = 1）

## 3. 核心算法

### 3.1 算子创建接口
```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo);
```
- **功能**: 创建 RoPE 算子描述符，验证输入张量形状并初始化元数据
- **关键步骤**:
  1. 将通用 handle 转换为 CPU 专用 handle
  2. 调用 `RoPEInfo::createRoPEInfo()` 创建并验证元数据
  3. 使用 `new` 分配 Descriptor 对象（workspace_size=0, opaque=nullptr）
- **复杂度**: O(1)（仅执行元数据验证和对象创建）

### 3.2 RoPE 计算核心
```cpp
template <typename Tdata, typename Tindex>
infiniStatus_t calculateRoPE(
    const RoPEInfo &info,
    Tdata *y,
    const Tdata *x,
    const Tindex *pos_ids,
    const Tdata *sin_table,
    const Tdata *cos_table);
```
- **功能**: 执行 RoPE 变换的核心计算逻辑
- **算法原理**:
  对于输入向量的每一对元素 `(x0, x1)`，根据位置编码表进行旋转：
  ```
  y0 = x0 * cos(θ) - x1 * sin(θ)
  y1 = x0 * sin(θ) + x1 * cos(θ)
  ```
  这是 2D 旋转矩阵的标准形式，θ 根据位置 ID 从预计算的 sin/cos 表中查找

- **并行策略**:
  - 使用 OpenMP 的 `#pragma omp parallel for` 并行化 batch 和 head 维度
  - 移除了 `collapse` 子句，确保外层循环（batch）被有效并行化
  - 内层循环（token 和维度）保持串行，避免过小的并行粒度

- **内存布局处理**:
  - **3D 张量** `[seqlen, nhead, dhead]`: batch stride 设为 0，逻辑上 batch=1
  - **4D 张量** `[batch, seqlen, nhead, dhead]`: 正常计算 batch 偏移
  - **位置 ID**:
    - 1D `[seqlen]`: 所有 batch 共享，`pos_offset = tok`
    - 2D `[batch, seqlen]`: 每个 batch 独立，`pos_offset = b * seqlen + tok`

- **算法变体**:
  - **GPT-J 风格** (`INFINIOP_ROPE_ALGO_GPT_J`): 交错配对
    ```
    pos0 = 2 * i, pos1 = 2 * i + 1
    ```
    例如：维度 [0,1,2,3,4,5] 处理为对 (0,1), (2,3), (4,5)

  - **GPT-NeoX 风格** (`INFINIOP_ROPE_ALGO_GPT_NEOX`): 前后半分离
    ```
    pos0 = i, pos1 = i + table_dim
    ```
    例如：dhead=8, table_dim=4 时，处理为对 (0,4), (1,5), (2,6), (3,7)

- **数据类型处理**:
  - **半精度类型** (fp16_t, bf16_t): 使用 `utils::cast<float>` 先转换为 float 计算，再转回原类型
    ```cpp
    float x0 = utils::cast<float>(x[x_offset + pos0]);
    float x1 = utils::cast<float>(x[x_offset + pos1]);
    float sin__ = utils::cast<float>(sin_table[table_offset + i]);
    float cos__ = utils::cast<float>(cos_table[table_offset + i]);
    y[y_offset + pos0] = utils::cast<Tdata>(x0 * cos__ - x1 * sin__);
    ```
  - **原生浮点类型** (float, double): 直接使用原生类型计算，避免转换开销

- **复杂度分析**:
  - 时间复杂度: O(batch × nhead × seqlen × table_dim)
  - 空间复杂度: O(1)（原地修改，无额外分配）
  - 并行度: O(batch × nhead)

### 3.3 类型分发器
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const;
```
- **功能**: 根据运行时数据类型和位置 ID 类型，实例化正确的模板函数
- **分发策略**:
  - 外层 `switch` 分发数据类型（fp16, bf16, f32, f64）
  - 内层宏 `ROPE_TYPE` 分发位置 ID 类型（所有整数类型）
  - 通过宏展开生成 4 × 8 = 32 种特化版本

- **实现细节**:
  ```cpp
  #define CALCULATE_ROPE(TDATA, TINDEX) \
      calculateRoPE(_info, (TDATA *)y, (const TDATA *)x, \
                    (const TINDEX *)pos_ids, \
                    (const TDATA *)sin_table, (const TDATA *)cos_table)

  #define ROPE_TYPE(TDATA)                        \
      switch (_info.pos_type) {                   \
      case INFINI_DTYPE_U8:                       \
          return CALCULATE_ROPE(TDATA, uint8_t);  \
      case INFINI_DTYPE_U16:                      \
          return CALCULATE_ROPE(TDATA, uint16_t); \
      // ... 其他 6 种整数类型
      }
  ```

## 4. 使用示例

```cpp
// 1. 准备张量描述符
// 输入: [batch=2, seqlen=128, nhead=32, dhead=128]
infiniopTensorDescriptor_t x_desc, y_desc, pos_desc, sin_desc, cos_desc;
// ... (创建描述符的代码省略)

// 2. 创建 RoPE 算子描述符
infiniopRoPEDescriptor_t rope_desc;
infiniStatus_t status = infiniopCreateRoPEDescriptor(
    handle,                    // CPU handle
    &rope_desc,
    y_desc,                    // 输出描述符
    x_desc,                    // 输入描述符
    pos_desc,                  // 位置 ID (可为 1D 或 2D)
    sin_desc,                  // sin 表 [max_pos, dhead/2]
    cos_desc,                  // cos 表 [max_pos, dhead/2]
    INFINIOP_ROPE_ALGO_GPT_NEOX  // 使用 GPT-NeoX 风格
);

// 3. 获取工作空间大小（CPU 实现为 0）
size_t workspace_size;
infiniopGetRoPEWorkspaceSize(rope_desc, &workspace_size);

// 4. 准备数据指针
void *x = ...;        // 输入张量数据
void *y = ...;        // 输出张量数据
void *pos_ids = ...;  // 位置 ID (如 [0,1,2,...,127] 或自定义)
void *sin_table = ...; // 预计算的 sin 表
void *cos_table = ...; // 预计算的 cos 表

// 5. 执行 RoPE 计算
status = infiniopRoPE(
    rope_desc,
    nullptr,             // workspace (CPU 不需要)
    0,                   // workspace_size
    y,                   // 输出
    x,                   // 输入
    pos_ids,             // 位置编码
    sin_table,           // sin 表
    cos_table,           // cos 表
    nullptr              // stream (CPU 不需要)
);

// 6. 清理资源
infiniopDestroyRoPEDescriptor(rope_desc);
```

## 5. 实现细节

### 5.1 内存管理
- **无动态分配**: 核心计算函数 `calculateRoPE` 中无任何 `new`/`malloc` 调用
- **栈上计算**: 所有临时变量（偏移量、转换后的浮点值）均在栈上分配
- **指针重解释**: 使用 `reinterpret_cast` 将通用 `void*` 转换为具体类型指针，零拷贝

### 5.2 并发控制
- **并行模型**: OpenMP fork-join 模型
- **同步点**: 并行区域结束隐式屏障（`#pragma omp parallel for` 默认行为）
- **数据竞争避免**:
  - 不同 batch-head 对写入 `y` 的不同内存区域
  - `sin_table`/`cos_table`/`x`/`pos_ids` 只读，无竞争
- **线程安全**: 完全线程安全，无共享可变状态

### 5.3 性能优化
- **模板特化**: 编译期为每种数据类型生成专用代码，避免运行时分支
- **半精度优化**: fp16/bf16 仅在必要时转换，计算使用 float 保持精度
- **缓存友好**: 内层循环按 `table_dim` 遍历，顺序访问 `sin_table`/`cos_table`
- **步长预计算**: 所有张量步长在 `RoPEInfo` 中预计算，运行时直接使用

### 5.4 错误处理
- **错误传播**: 使用 `CHECK_RESULT(info)` 宏检查 `RoPEInfo` 创建失败
- **错误码**: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE` 表示不支持的数据类型
- **断言失败**: 编译期 `static_assert` 确保模板类型有效（通过 `constexpr if`）

### 5.5 设计模式
- **CRTP (奇异递归模板模式)**: `DESCRIPTOR(cpu)` 宏在 `op::rope::cpu` 命名空间中实例化 Descriptor 类
- **工厂方法**: `Descriptor::create()` 作为静态工厂，封装对象创建逻辑
- **策略模式**: 通过 `algo` 参数选择 GPT-J 或 GPT-NeoX 算法变体
- **类型擦除**: 公共 API 使用 `void*`，内部通过模板恢复类型信息

### 5.6 依赖关系
- **内部依赖**:
  - `../rope.h`: 提供 `RoPEInfo` 类和 `DESCRIPTOR` 宏定义
  - `../../../devices/cpu/common_cpu.h`: 提供 CPU Handle 类型定义
  - `../../../utils/custom_types.h`: 提供 `fp16_t`/`bf16_t` 和 `utils::cast` 转换函数
- **外部依赖**:
  - OpenMP (可选): 通过 `#ifdef ENABLE_OMP` 条件编译
  - C++ 标准库: `<type_traits>` (用于 `std::is_same`)

### 5.7 算法选择指南
- **GPT-J 风格**: 适用于注意力头内部维度交错的模型（如原始 GPT-J）
- **GPT-NeoX 风格**: 适用于维度分半的模型（如 LLaMA、GPT-NeoX），更常见的现代实现
- **性能差异**: 两种算法计算量相同，仅内存访问模式不同
  - GPT-J: 访问 `x[2*i]` 和 `x[2*i+1]`，相邻内存
  - GPT-NeoX: 访问 `x[i]` 和 `x[i+table_dim]`，跨度为 `table_dim`

## 6. 性能特性

- **理论峰值**: 受限于内存带宽，计算密度低（主要是乘加和三角函数查表）
- **并行扩展性**: 在 batch×nhead ≥ 物理核心数时线性扩展
- **内存访问模式**:
  - `x`/`y`: 随机访问（取决于步长，但最后一维保证连续）
  - `sin_table`/`cos_table`: 顺序访问，缓存友好
  - `pos_ids`: 低频访问（每个 token 一次）
- **适用场景**: 适合中小型 batch 和序列长度（CPU 内存带宽优势），大规模推荐使用 GPU 后端
