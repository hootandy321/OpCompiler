# 目录: InfiniCore/src/infiniop/sort 架构全景

## 1. 子系统职责

`sort` 目录是 InfiniOp 算子库中的**排序与堆操作模块**，专注于在异构加速设备（特别是昆仑 XPU）上实现高效的堆数据结构操作。该模块提供了设备端（device-side）的堆算法模板，支持最小堆和最大堆的构建、更新与排序操作，是 Top-K、优先队列、归并排序等高级算法的基础组件。

在 InfiniOp 整体架构中，sort 模块作为底层算子支撑层，为上层的张量操作、注意力机制、采样等计算密集型任务提供高性能的堆排序原语。

## 2. 模块导航

* **kunlun**:
    * *功能*: 提供昆仑 XPU 设备端的堆操作模板函数库
    * *职责*: 实现共享内存和局部内存两种内存模式下的最小堆/最大堆构建、更新与排序算法，支持键值对（Key-Value）操作的通用模板

## 3. 架构逻辑图解

### 3.1 内存层级设计

该模块采用双内存层级架构，针对不同的并行计算场景优化：

1. **共享内存堆（Shared Memory Heap, `sm_` 前缀）**：
   * 适用于线程块内协作计算
   * 使用 `_shared_ptr_` 修饰符，确保线程间数据共享
   * 典型应用场景：Block 内的 Top-K 选择、局部归并

2. **局部内存堆（Local Memory Heap, `lm_` 前缀）**：
   * 适用于单线程独立计算
   * 直接使用指针访问，无跨线程同步
   * 典型应用场景：单线程维护小规模有序结构

### 3.2 核心算法组件

模块提供 4 组完整的堆操作模板，每组包含 3 个核心函数：

**最小堆系列**：
```
make_sm_min_heap / make_lm_min_heap
    └─> 从无序数组构建最小堆（O(n)）

update_sm_min_heap / update_lm_min_heap
    └─> 从指定节点向下调整堆（O(log n)）

sort_sm_min_heap / sort_lm_min_heap
    └─> 堆排序输出递增序列（O(n log n)）
```

**最大堆系列**：
```
make_sm_max_heap / make_lm_max_heap
    └─> 从无序数组构建最大堆

update_sm_max_heap / update_lm_max_heap
    └─> 从指定节点向下调整堆

sort_sm_max_heap / sort_lm_max_heap
    └─> 堆排序输出递减序列
```

### 3.3 数据流与依赖关系

```
输入数据 (Key-Value Pairs)
        │
        ▼
   [make_*_heap]  ──>  初始化堆结构
        │
        ▼
   应用场景分支：
   │
   ├─> Top-K 选择: 重复 [update_*_heap] 替换堆顶
   ├─> 优先队列: 插入元素后调用 [update_*_heap]
   └─> 完整排序: 直接调用 [sort_*_heap]
        │
        ▼
   输出有序序列
```

### 3.4 模板泛化设计

所有函数均采用双模板参数 `<typename TK, typename TV>`：
- **TK (Key Type)**: 排序键类型（如 float、int）
- **TV (Value Type)**: 伴随值类型（如索引、向量数据）

这种设计支持"按键排序，值随键动"的语义，典型应用如：
- 按分数排序，返回原始索引
- 按距离排序，返回坐标点

### 3.5 辅助工具函数

**类型转换层**：
- `primitive_cast<TX, TY>`: 针对昆仑 XPU 的 SIMD 优化类型转换
  - `float → int`: 使用 `vfloat2fix.rz` 指令向零取整
  - `int → float`: 使用 `vfix2float.rn` 指令最近舍入
  - `float → float`: 使用 32 向量宽度的内存拷贝优化

**并行计算工具**：
- `partition()`: 将数据区间按线程数均衡分块
  - 采用 `roundup_div` 实现块对齐
  - 处理余数块，确保负载均衡
- `roundup_div_p()`: 向上取整除法
- `min_p()`: 最小值比较

### 3.6 硬件适配特性

该实现针对昆仑 XPU 硬件特性深度优化：

1. **SIMD 指令集**：
   - 使用 `vload_lm_float32x16` / `vstore_lm_float32x16` 进行 16 元素向量化加载/存储
   - 利用内联汇编 `__asm__ __volatile__` 直接调用 XPU 指令（`vr0` 寄存器操作）

2. **内存屏障**：
   - `mfence_lm()` 确保局部内存操作一致性

3. **类型系统扩展**：
   - 使用 `float32x16_t`、`int32x16_t` 等 SIMD 向量类型
   - 通过 `_shared_ptr_` 区分共享内存地址空间

### 3.7 典型应用场景

在 InfiniOp 生态系统中，该堆操作模块主要支撑以下高层功能：

1. **Top-K 采样**：在生成式推理中，从词汇表中选择概率最高的 K 个 token
2. **注意力机制**：对注意力分数进行排序，实现稀疏注意力模式
3. **张量排序**：为 `infiniop::sort` 算子提供底层堆原语
4. **归并网络**：在多路归并中使用堆维护最小/最大元素

### 3.8 扩展性与限制

**当前状态**：
- 仅实现 `kunlun` 后端，其他硬件（CUDA、Ascend 等）缺失实现
- 纯头文件设计，无独立的编译单元

**潜在扩展方向**：
- 复用相同的模板接口，为其他硬件平台实现后端
- 增加堆合并（heap merge）、堆化（heapify）等高级操作
- 支持外部排序（外存堆）以处理超大规模数据

## 4. 技术规格总结

| 属性 | 描述 |
|------|------|
| **语言** | C++ (CUDA-style device code) |
| **依赖** | 昆仑 XPU SDK (`xpu/kernel/xtdk_simd_xpu2.h`) |
| **内存模型** | Shared Memory + Local Memory |
| **线程安全** | 块内共享内存需同步，局部内存天然线程安全 |
| **复杂度** | 构建 O(n)，更新 O(log n)，排序 O(n log n) |
| **模板参数** | TK (键类型), TV (值类型) |
| **后端** | Kunlun XPU |

## 5. 文档状态说明

- `kunlun/` 子目录无独立文档，本分析基于源代码 `heap.h` 生成
- 该模块为叶子节点（Leaf Node），包含完整的实现代码
- 暂未发现单元测试文档或性能基准报告
