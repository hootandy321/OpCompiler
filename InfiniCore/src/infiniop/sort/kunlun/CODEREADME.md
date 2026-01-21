# Kunlun XPU Heap Operations Core Implementation Documentation

Kunlun XPU (昆仑芯片) 架构下的高性能堆排序与键值对操作实现。该模块提供了共享内存(shared memory)和本地内存(local memory)两种内存层次下的最小堆/最大堆操作，以及针对昆仑 XPU2 SIMD 指令集优化的类型转换和向量内存操作。

## 1. Module Structure

- **`heap.h`**: Kunlun XPU 设备端堆操作核心实现，包含堆的构建、更新、排序算法，以及 SIMD 优化的类型转换函数

## 2. Core Functions

### 共享内存堆操作 (Shared Memory Heap Operations)

#### `sm_swap_kv<TK, TV>`
- **Location**: `heap.h:6-14`
- **Primary Function**: 交换共享内存中的键值对
- **Key Parameters**:
  - `k0, v0`: 第一个键值对的指针（_shared_ptr_ 修饰）
  - `k1, v1`: 第二个键值对的指针（_shared_ptr_ 修饰）
- **Implementation**: 使用临时变量执行原子交换操作，确保键和值的同步交换

#### `update_sm_min_heap<TK, TV>`
- **Location**: `heap.h:17-38`
- **Primary Function**: 从指定索引开始向下调整最小堆，维护堆性质
- **Algorithm**: 标准堆下沉(sift-down)算法
  - 计算当前节点的左右子节点索引（2*idx+1 和 2*idx+2）
  - 比较左右子节点值，选择较小者（最小堆性质）
  - 如果父节点大于较小子节点，交换并继续下沉
  - 时间复杂度: O(log n)，其中 n 为堆容量
- **Key Parameters**:
  - `heap_key`: 共享内存中的键数组
  - `heap_value`: 共享内存中的值数组
  - `idx`: 开始调整的节点索引
  - `heap_capacity`: 堆的有效容量
- **Termination Conditions**:
  - 当前节点为叶子节点（无子节点）
  - 当前节点值已满足堆性质（最小堆中父节点 ≤ 子节点）

#### `make_sm_min_heap<TK, TV>`
- **Location**: `heap.h:41-46`
- **Primary Function**: 将共享内存中的无序数组构建为最小堆
- **Algorithm**: Floyd 堆构造算法（自底向上）
  - 从最后一个非叶子节点开始（索引 size/2-1）
  - 依次对每个节点调用 `update_sm_min_heap` 进行下沉调整
  - 时间复杂度: O(n)，优于逐个插入的 O(n log n)

#### `sort_sm_min_heap<TK, TV>`
- **Location**: `heap.h:49-55`
- **Primary Function**: 对最小堆进行原地排序，最终得到降序数组
- **Algorithm**: 堆排序
  - 反复将堆顶（最小元素）与堆末尾元素交换
  - 缩小堆容量并重新调整堆
  - 时间复杂度: O(n log n)
  - 空间复杂度: O(1)，原地排序
- **Result**: 堆内元素按降序排列（最大值在前）

#### `update_sm_max_heap<TK, TV>`
- **Location**: `heap.h:58-79`
- **Primary Function**: 从指定索引开始向下调整最大堆
- **Algorithm**: 与 `update_sm_min_heap` 对称，选择较大子节点进行交换
  - 选择左右子节点中较大者
  - 如果父节点小于较大子节点，交换并继续下沉
  - 时间复杂度: O(log n)

#### `make_sm_max_heap<TK, TV>`
- **Location**: `heap.h:82-87`
- **Primary Function**: 将共享内存中的无序数组构建为最大堆
- **Algorithm**: Floyd 算法，自底向上构建最大堆

#### `sort_sm_max_heap<TK, TV>`
- **Location**: `heap.h:90-96`
- **Primary Function**: 对最大堆进行原地排序，最终得到升序数组
- **Result**: 堆内元素按升序排列（最小值在前）

### 本地内存堆操作 (Local Memory Heap Operations)

#### `lm_swap_kv<TK, TV>`
- **Location**: `heap.h:99-107`
- **Primary Function**: 交换本地内存中的键值对
- **Difference**: 与 `sm_swap_kv` 功能相同，但操作本地内存而非共享内存

#### `update_lm_min_heap<TK, TV>`, `make_lm_min_heap<TK, TV>`, `sort_lm_min_heap<TK, TV>`
- **Location**: `heap.h:110-146`
- **Primary Function**: 本地内存版本的最小堆操作
- **Implementation**: 逻辑与共享内存版本完全一致，仅内存空间不同

#### `update_lm_max_heap<TK, TV>`, `make_lm_max_heap<TK, TV>`, `sort_lm_max_heap<TK, TV>`
- **Location**: `heap.h:149-185`
- **Primary Function**: 本地内存版本的最大堆操作

### 工具函数 (Utility Functions)

#### `roundup_div_p<TID>`
- **Location**: `heap.h:188-190`
- **Primary Function**: 向上取整除法
- **Formula**: `(a + b - 1) / b`
- **Use Case**: 计算分块任务时确定需要的块数

#### `min_p<T>`
- **Location**: `heap.h:193-195`
- **Primary Function**: 返回两个值中的较小者
- **Implementation**: 三元运算符 `a < b ? a : b`

#### `partition<TID>`
- **Location**: `heap.h:198-205`
- **Primary Function**: 将长度为 `len` 的数据按 `align` 对齐后，均匀分区给 `nthreads` 个线程
- **Algorithm**:
  1. 计算对齐后的总块数: `block_cnt = roundup_div(len, align)`
  2. 计算余数块: `remain_block = block_cnt % nthreads`
  3. 使用负载均衡策略分配：
     - 前面 `remain_block` 个线程各多分配一个块
     - 其余线程平均分配剩余块
  4. 将块索引转换为字节索引，并限制在 `len` 范围内
- **Load Balancing**: 采用块级负载均衡，避免线程间负载不均
- **Output Parameters**:
  - `start`: 当前线程负责的起始索引
  - `end`: 当前线程负责的结束索引（不包含）

### 类型转换与 SIMD 优化 (Type Casting and SIMD)

#### `primitive_cast<TX, TY>` (通用模板)
- **Location**: `heap.h:207-210`
- **Primary Function**: 空操作的通用模板实现
- **Use Case**: 为未特化的类型组合提供默认行为

#### `primitive_cast<float, int>` (特化)
- **Location**: `heap.h:212-224`
- **Primary Function**: 使用昆仑 XPU2 SIMD 指令将 float 数组转换为 int 数组（向零截断）
- **Implementation Details**:
  - 使用 `float32x16_t` 向量类型，每次处理 16 个元素
  - 内联汇编指令 `vfloat2fix.rz` 执行向零截断的浮点转定点
  - `vstore_mask16.mz` 掩码存储指令，处理边界情况
  - `mfence_lm()` 本地内存栅栏，确保写入完成
- **SIMD Width**: 16-way SIMD，单次处理 512 位（16 × 32-bit）
- **Rounding Mode**: `rz` (round toward zero)

#### `primitive_cast<int, float>` (特化)
- **Location**: `heap.h:225-237`
- **Primary Function**: 使用昆仑 XPU2 SIMD 指令将 int 数组转换为 float 数组
- **Implementation Details**:
  - 使用 `int32x16_t` 向量类型
  - 内联汇编指令 `vfix2float.rn` 执行定点转浮点（最近偶数舍入）
  - 相同的掩码存储和内存栅栏机制
- **Rounding Mode**: `rn` (round to nearest, ties to even)

#### `primitive_cast<float, float>` (特化)
- **Location**: `heap.h:249-262`
- **Primary Function**: float 到 float 的拷贝（或原地空操作）
- **Optimization**:
  - 检查源地址和目标地址是否相同，相同则直接返回（避免不必要的拷贝）
  - 使用双向量加载/存储（每次 32 个元素）提高带宽利用率
  - 32 元素分块: 两个 `float32x16_t` 向量并行处理

#### `vload2_lm<float>`
- **Location**: `heap.h:239-242`
- **Primary Function**: 从本地内存加载连续 32 个 float 到两个向量寄存器
- **Implementation**: 使用 `__builtin_xpu2_vload_mask16_mr1` 内置函数

#### `vstore2_lm<float>`
- **Location**: `heap.h:244-247`
- **Primary Function**: 将两个向量寄存器存储连续 32 个 float 到本地内存
- **Implementation**: 使用 `vstore_lm_float32x16` 标准向量存储

## 3. API Interface

```cpp
// 共享内存堆操作 API
template <typename TK, typename TV>
__device__ void update_sm_min_heap(_shared_ptr_ TK *heap_key,
                                   _shared_ptr_ TV *heap_value,
                                   int idx, int heap_capacity);
// 在共享内存中维护最小堆性质，从索引 idx 开始下沉调整

template <typename TK, typename TV>
__device__ void make_sm_min_heap(_shared_ptr_ TK *heap_key,
                                 _shared_ptr_ TV *heap_value,
                                 int size);
// 将共享内存中的键值对数组构建为最小堆

template <typename TK, typename TV>
__device__ void sort_sm_min_heap(_shared_ptr_ TK *heap_key,
                                 _shared_ptr_ TV *heap_value,
                                 int heap_capacity);
// 对共享内存中的最小堆进行排序，结果为降序数组

template <typename TK, typename TV>
__device__ void update_sm_max_heap(_shared_ptr_ TK *heap_key,
                                   _shared_ptr_ TV *heap_value,
                                   int idx, int heap_capacity);
// 在共享内存中维护最大堆性质，从索引 idx 开始下沉调整

template <typename TK, typename TV>
__device__ void make_sm_max_heap(_shared_ptr_ TK *heap_key,
                                 _shared_ptr_ TV *heap_value,
                                 int size);
// 将共享内存中的键值对数组构建为最大堆

template <typename TK, typename TV>
__device__ void sort_sm_max_heap(_shared_ptr_ TK *heap_key,
                                 _shared_ptr_ TV *heap_value,
                                 int heap_capacity);
// 对共享内存中的最大堆进行排序，结果为升序数组

// 本地内存堆操作 API（签名与共享内存版本类似，但无 _shared_ptr_ 修饰）
template <typename TK, typename TV>
__device__ void update_lm_min_heap(TK *heap_key, TV *heap_value,
                                   int idx, int heap_capacity);

template <typename TK, typename TV>
__device__ void make_lm_min_heap(TK *heap_key, TV *heap_value, int size);

template <typename TK, typename TV>
__device__ void sort_lm_min_heap(TK *heap_key, TV *heap_value, int heap_capacity);

template <typename TK, typename TV>
__device__ void update_lm_max_heap(TK *heap_key, TV *heap_value,
                                   int idx, int heap_capacity);

template <typename TK, typename TV>
__device__ void make_lm_max_heap(TK *heap_key, TV *heap_value, int size);

template <typename TK, typename TV>
__device__ void sort_lm_max_heap(TK *heap_key, TV *heap_value, int heap_capacity);

// 工具函数 API
template <typename TID>
__device__ TID roundup_div_p(TID a, TID b);
// 向上取整除法

template <typename T>
__device__ T min_p(T a, T b);
// 返回较小值

template <typename TID>
__device__ void partition(int tid, int nthreads, TID len, int align,
                         TID *start, TID *end);
// 将数据负载均衡地分区给多个线程

// SIMD 类型转换 API
template <typename TX, typename TY>
__device__ void primitive_cast(const TX *x, TY *y, int len);
// 通用类型转换（空操作模板）

template <>
__device__ void primitive_cast<float, int>(const float *x, int *y, int len);
// float → int 转换（向零截断），使用 16-way SIMD

template <>
__device__ void primitive_cast<int, float>(const int *x, float *y, int len);
// int → float 转换（最近偶数舍入），使用 16-way SIMD

template <>
__device__ void primitive_cast<float, float>(const float *x, float *y, int len);
// float 拷贝，使用 32 元素/次的双向量化优化
```

## 4. Usage Example

```cpp
#include "xpu/kernel/xtdk_simd_xpu2.h"
#include "heap.h"

__global__ void topk_kernel(const float *input_keys, const int *input_values,
                            float *output_keys, int *output_values,
                            int k, int n) {
    // 1. 在共享内存中分配堆空间
    __shared__ float sm_keys[1024];
    __shared__ int sm_values[1024];

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // 2. 负载均衡地分区输入数据
    int start, end;
    partition(tid, nthreads, n, 1, &start, &end);

    // 3. 每个线程处理自己的分区，构建局部最大堆
    // 使用本地内存存储临时堆
    float lm_keys[256];
    int lm_values[256];
    int heap_size = 0;

    for (int i = start; i < end; i++) {
        if (heap_size < 256) {
            lm_keys[heap_size] = input_keys[i];
            lm_values[heap_size] = input_values[i];
            heap_size++;
        } else {
            // 堆已满，如果新元素大于堆顶（最小元素），替换并调整
            if (input_keys[i] > lm_keys[0]) {
                lm_keys[0] = input_keys[i];
                lm_values[0] = input_values[i];
                update_lm_min_heap(lm_keys, lm_values, 0, heap_size);
            }
        }
    }

    // 4. 将本地堆拷贝到共享内存（使用 SIMD 优化的拷贝）
    __syncthreads();
    primitive_cast(lm_keys, sm_keys + tid * 256, heap_size);
    primitive_cast(lm_values, sm_values + tid * 256, heap_size);

    __syncthreads();

    // 5. 合并所有线程的局部堆到全局堆
    if (tid == 0) {
        int total_size = min_p(n, 256 * nthreads);
        make_sm_max_heap(sm_keys, sm_values, total_size);

        // 提取 Top-K 元素
        for (int i = 0; i < k && i < total_size; i++) {
            output_keys[i] = sm_keys[0];
            output_values[i] = sm_values[0];

            // 将堆顶（最大元素）与堆末尾交换并缩小堆
            sm_swap_kv(&sm_keys[0], &sm_values[0],
                      &sm_keys[total_size - 1 - i],
                      &sm_values[total_size - 1 - i]);
            update_sm_max_heap(sm_keys, sm_values, 0, total_size - 1 - i);
        }
    }
}

// 调用示例
void launch_topk(const float *d_keys, const int *d_values,
                 float *d_result_keys, int *d_result_values,
                 int k, int n) {
    int block_size = 256;
    int grid_size = 1;
    topk_kernel<<<grid_size, block_size>>>(d_keys, d_values,
                                           d_result_keys, d_result_values,
                                           k, n);
}
```

## 5. Implementation Details

### 内存层次优化 (Memory Hierarchy Optimization)
- **共享内存堆 (sm_)**: 用于线程间协作，支持块内数据共享。所有线程可访问同一块共享内存，适用于需要线程间通信的场景（如归约、合并排序）。
- **本地内存堆 (lm_)**: 线程私有内存，访问延迟最低。每个线程独立维护堆，无竞争，适合线程局部处理（如 Top-K 局部筛选）。
- **双版本设计**: 提供共享内存和本地内存两套完整 API，允许在不同场景下选择最优内存空间。

### 堆算法实现细节 (Heap Algorithm Details)
- **二叉堆存储**: 使用数组实现完全二叉堆，索引映射：
  - 父节点: `(idx - 1) / 2`
  - 左子节点: `idx * 2 + 1`
  - 右子节点: `idx * 2 + 2`
- **下沉调整 (Sift-Down)**: 从给定节点向下调整，时间复杂度 O(log n)。核心优化：
  - 提前计算左右子节点索引
  - 使用比较结果（0 或 1）直接索引较小/较大子节点，避免分支
  - 子节点越界检查：优先检查右子节点，左子节点不存在时直接终止
- **Floyd 堆构造**: 自底向上构建，从最后一个非叶子节点（`size/2 - 1`）开始向前遍历，线性时间复杂度 O(n)。
- **堆排序**: 原地排序，反复交换堆顶与堆末尾并缩小堆。空间复杂度 O(1)，时间复杂度 O(n log n)。

### 昆仑 XPU2 SIMD 优化 (Kunlun XPU2 SIMD Optimization)
- **SIMD 宽度**: 16-way（512 位向量），每条指令处理 16 个 32-bit 元素。
- **向量类型**:
  - `float32x16_t`: 16 个 float32 向量
  - `int32x16_t`: 16 个 int32 向量
- **内联汇编指令**:
  - `vfloat2fix.rz`: 浮点转定点，向零截断（round toward zero），用于 float → int
  - `vfix2float.rn`: 定点转浮点，最近偶数舍入（round to nearest, ties to even），用于 int → float
  - `vstore_mask16.mz`: 掩码存储指令，`mz` 后缀表示掩码模式（masked zero），处理非对齐或边界情况
- **双向量流水线**: `primitive_cast<float, float>` 使用双向量（32 元素）加载/存储，隐藏内存延迟，提高带宽利用率。
- **内存栅栏**: `mfence_lm()` 本地内存栅栏确保 SIMD 写操作完成后才执行后续指令，避免内存一致性问题。

### 负载均衡策略 (Load Balancing Strategy)
- **块级分区**: `partition` 函数将数据划分为对齐块（`align` 参数），避免跨块访问。
- **余数块分配**: 总块数如果不能被线程数整除，余数块分配给前几个线程，每个线程最多相差一个块。
- **公式推导**:
  - 基础块分配: `base_blocks = block_cnt / nthreads`
  - 余数块: `remain_blocks = block_cnt % nthreads`
  - 线程 `tid` 的起始块: `start_block = base_blocks * tid + min(tid, remain_blocks)`
  - 线程 `tid` 的结束块: `end_block = start_block + base_blocks + (tid < remain_blocks)`
- **应用场景**: 并行排序、Top-K、规约等需要均匀分配任务的场景。

### 模板元编程 (Template Metaprogramming)
- **双模板参数**: `<TK, TV>` 支持键值对类型独立，例如 `<float, int>`、`<int, float>`、`<double, long>` 等。
- **类型转换特化**: `primitive_cast` 对特定类型组合（如 `float → int`）提供特化实现，未特化组合回退到空操作模板。
- **设备端函数**: 所有函数标记为 `__device__`，仅可在 Kunlun XPU 设备端调用，不可在主机端直接使用。
- **内联优化**: `static __device__ inline` 确保函数在编译时内联，减少调用开销，适合高频调用的堆操作。

### 性能复杂度总结 (Performance Complexity Summary)
| 操作 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| `update_*_heap` | O(log n) | O(1) | 堆插入、删除后的调整 |
| `make_*_heap` | O(n) | O(1) | 批量构建堆 |
| `sort_*_heap` | O(n log n) | O(1) | 原地排序 |
| `partition` | O(1) | O(1) | 任务负载均衡 |
| `primitive_cast` (SIMD) | O(n/16) | O(1) | 类型转换/拷贝 |

### 依赖项 (Dependencies)
- **昆仑 XPU SDK**: `xpu/kernel/xtdk_simd_xpu2.h`，提供昆仑 XPU2 SIMD 内置函数和向量类型
- **CUDA 兼容语法**: 使用 `_shared_ptr_`、`__device__`、`__global__` 等 CUDA 风格修饰符（昆仑 XPU 编译器支持）
- **内联汇编**: 部分核心指令直接使用内联汇编编写，确保最优性能

### 设计模式 (Design Patterns)
- **策略模式**: 提供最小堆和最大堆两种策略，通过不同的比较逻辑实现
- **模板方法模式**: 堆操作的骨架相同（下沉、构建、排序），通过模板参数适配不同类型和内存空间
- **RAII (Resource Acquisition Is Initialization)**: 不适用（无对象生命周期管理）
- **函数式编程**: 大量使用纯函数（无副作用），便于编译器优化和 SIMD 向量化
