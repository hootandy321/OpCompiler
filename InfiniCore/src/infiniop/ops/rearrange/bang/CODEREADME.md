# Rearrange Bang 算子核心实现文档

该模块实现了 Infini 框架中针对寒武纪 MLU (Cambricon Machine Learning Unit) 硬件的张量重排操作（Rearrange Operator）。Rearrange 操作是一种通用的内存重排原语，支持任意维度的张量转置、重塑、广播等操作。本实现通过 BANG (BANG Advanced Neural Network) 语言编写 MLU 核函数，并针对 MLU 硬件特性进行了深度优化。

## 1. 模块结构

- **`rearrange_bang.h`**: 声明文件，通过宏定义生成 `Descriptor` 类的基本结构，继承自 `InfiniopDescriptor` 基类
- **`rearrange_bang.mlu`**: 核心实现文件，包含两个 MLU 核函数（通用 rearrange 和优化版 rearrange2d）以及 Descriptor 的完整实现

## 2. 核心类

### `Descriptor::Opaque`
- **位置**: `rearrange_bang.mlu` (第 7-18 行)
- **主要功能**: 存储算子执行所需的运行时元数据和设备侧资源，作为 `Descriptor` 的内部不透明数据结构
- **关键成员**:
  - `meta`: `utils::RearrangeMeta` 类型，存储重排操作的元数据（步幅、维度、元素总数等）
  - `element_size`: `size_t` 类型，单个数据元素的字节大小（如 float 为 4，int16_t 为 2）
  - `d_idx_strides`: `int*` 设备指针，索引步幅数组，用于将线性索引映射到多维坐标
  - `d_dst_strides`: `int*` 设备指针，目标张量的步幅数组
  - `d_src_strides`: `int*` 设备指针，源张量的步幅数组
  - `use_2d_copy`: `bool` 标志，指示是否使用优化的 2D 拷贝内核（针对纯转置场景）
  - `outer_dim`: `int`，2D 拷贝时的外维度大小（行数）
  - `inner_dim`: `int`，2D 拷贝时的内维度大小（列数）
  - `dst_stride`: `int`，2D 拷贝时目标张量的行跨距（字节为单位）
  - `src_stride`: `int`，2D 拷贝时源张量的行跨距（字节为单位）

### `Descriptor`
- **位置**: 由宏 `DESCRIPTOR(bang)` 在 `rearrange_bang.h` 中定义，实现在 `rearrange_bang.mlu`
- **主要功能**: 封装 Rearrange 算子的完整生命周期管理，包括创建、执行和资源清理
- **关键成员**:
  - `_opaque`: `Opaque*` 指针，指向不透明的运行时数据
  - `_meta`: `utils::RearrangeMeta` 类型，存储重排元数据（继承自宏定义）
- **核心方法**:
  - `create(handle_, desc_ptr, y_desc, x_desc)` (第 136-253 行): 构造函数，执行参数验证、元数据计算、设备内存分配和核函数选择
  - `calculate(y, x, stream)` (第 255-297 行): 执行函数，根据 `use_2d_copy` 标志选择启动 `rearrange` 或 `rearrange2d` 核函数
  - `~Descriptor()` (第 20-27 行): 析构函数，释放设备侧分配的步幅数组内存
- **生命周期**:
  1. 通过静态方法 `create` 创建，验证输入张量形状和数据类型一致性
  2. 计算源/目标张量的步幅信息，构建 `RearrangeMeta` 元数据
  3. 检测是否为 2D 纯转置场景，决定使用通用核函数还是优化核函数
  4. 分配设备内存并拷贝步幅数组（仅通用核函数需要）
  5. 通过 `calculate` 方法执行计算，在 MLU 队列上启动核函数
  6. 析构时释放设备侧资源

## 3. API 接口

```cpp
// 创建 Rearrange 算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // Infini 框架句柄（包含设备和上下文信息）
    Descriptor **desc_ptr,                 // 输出参数，返回创建的描述符指针
    infiniopTensorDescriptor_t y_desc,     // 目标张量描述符
    infiniopTensorDescriptor_t x_desc);    // 源张量描述符
// 返回 INFINI_STATUS_SUCCESS 表示成功，其他值表示错误码

// 执行 Rearrange 计算
infiniStatus_t Descriptor::calculate(
    void *y,                               // 目标张量的设备地址
    const void *x,                         // 源张量的设备地址
    void *stream) const;                   // MLU 计算队列（cnrtQueue_t）
// 返回 INFINI_STATUS_SUCCESS 表示成功，其他值表示错误码
```

### MLU 核函数接口

```cpp
// 通用重排核函数（支持任意维度和步幅模式）
__mlu_global__ void rearrange(
    char *dst,                    // 目标内存起始地址（字节指针）
    const char *src,              // 源内存起始地址（字节指针）
    const int *idx_strides,       // 索引步幅数组（设备内存）
    const int *dst_strides,       // 目标步幅数组（设备内存）
    const int *src_strides,       // 源步幅数组（设备内存）
    int ndim,                     // 张量维度数
    int count,                    // 总元素数量
    int unit_size);               // 单个元素的字节大小

// 优化的 2D 转置核函数（针对纯矩阵转置场景）
__mlu_global__ void rearrange2d(
    char *dst,                    // 目标内存起始地址
    const char *src,              // 源内存起始地址
    int outer_dim,                // 外维度大小（行数）
    int inner_dim,                // 内维度大小（列数）
    int dst_stride_bytes,         // 目标行跨距（字节）
    int src_stride_bytes,         // 源行跨距（字节）
    int unit_size);               // 单个元素的字节大小
```

## 4. 使用示例

```cpp
// 示例：在 MLU 上执行矩阵转置 (100 x 100 float 矩阵)
#include "rearrange_bang.h"

// 1. 创建 MLU 句柄
infiniopHandle_t handle;
device::bang::Handle::create(&handle, 0); // 使用设备 0

// 2. 定义张量形状和步幅
// 源张量：行主序 (row-major)，shape [100, 100]
// strides [1, 100] 表示最内层维度连续
size_t x_shape[] = {100, 100};
ptrdiff_t x_strides[] = {1, 100};

// 目标张量：列主序 (column-major)，shape [100, 100]
// strides [1, 100] 表示转置后的行主序
size_t y_shape[] = {100, 100};
ptrdiff_t y_strides[] = {1, 100};

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_FLOAT32, 2, x_shape, x_strides);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_FLOAT32, 2, y_shape, y_strides);

// 3. 创建算子描述符（会自动选择优化版 rearrange2d 核函数）
op::rearrange::bang::Descriptor *rearrange_desc;
infiniStatus_t status = op::rearrange::bang::Descriptor::create(
    handle, &rearrange_desc, y_desc, x_desc);

// 4. 分配 MLU 内存并初始化输入数据
void *d_x, *d_y;
size_t nbytes = 100 * 100 * sizeof(float);
cnrtMalloc(&d_x, nbytes);
cnrtMalloc(&d_y, nbytes);
// ... 通过 cnrtMemcpy 拷贝数据到 d_x ...

// 5. 创建计算队列并执行转置
cnrtQueue_t queue;
cnrtQueueCreate(&queue);
rearrange_desc->calculate(d_y, d_x, queue);
cnrtQueueSync(queue); // 等待计算完成

// 6. 拷贝结果回主机并清理资源
// ... cnrtMemcpy(d_y, host_y, ...) ...
cnrtFree(d_x);
cnrtFree(d_y);
cnrtQueueDestroy(queue);
delete rearrange_desc;
```

## 5. 实现细节

### 内存管理策略
- **设备内存分配**: 使用 `cnrtMalloc` 在 MLU 全局内存（GDRAM）上分配三个步幅数组（`d_idx_strides`, `d_dst_strides`, `d_src_strides`），每个数组的长度等于张量维度数（ndim），元素类型为 32 位整数
- **异步数据传输**: 在 `create` 函数中，使用 `cnrtMemcpyAsync` 配合专用队列将步幅数据从主机异步传输到设备，通过 `cnrtQueueSync` 确保传输完成
- **自动资源释放**: 析构函数中使用 `cnrtFree` 释放所有设备侧分配的内存，并通过 `delete` 释放 Opaque 对象本身，遵循 RAII (Resource Acquisition Is Initialization) 原则

### 并发执行与并行化
- **MLU 集群并行**: 核函数使用 `cnrtDim3_t` 配置执行维度，设置为 `dim.x=4, dim.y=10, dim.z=1`，表示使用 4 个 MLU 集群（clusters），每个集群 10 个任务单元，总共 40 个并行任务
- **函数类型**: 使用 `CNRT_FUNC_TYPE_UNION1` 标志，表示采用 Union1 类型的核函数启动配置，适合计算密集型任务
- **Chunk 分块策略**:
  - 通用 `rearrange` 核函数采用 "chunk-based" 并行策略，将总元素数（count）划分为固定大小（256）的块（chunks），每个任务处理多个连续的块
  - 计算公式：`chunks_per_task = (num_chunks + task_dim - 1) / task_dim`，确保负载均衡
  - 每个任务独立计算其负责的块范围 `[start_chunk, end_chunk)`

### 性能优化技术

#### 5.1 寄存器预取优化
- **局部数组缓存**: 在通用核函数中，使用固定大小的局部数组 `local_idx_strides[8]`, `local_dst_strides[8]`, `local_src_strides[8]` 将设备全局内存中的步幅数据预取到寄存器（假设维度数 ≤ 8）
- **编译器指令**: 使用 `#pragma unroll` 指示编译器展开步幅拷贝循环，减少分支开销和内存访问延迟
- **性能影响**: 避免在每次元素处理时重复访问全局内存，显著降低访存延迟

#### 5.2 分块处理
- **Chunk 大小调优**: 设置 `chunk_size = 256`，针对 MLU 的缓存行大小（cache lines）进行优化，提高数据局部性
- **边界处理**: 使用 `std::min(start + chunk_size, count)` 确保最后一个块不会越界
- **负载均衡**: 通过整数除法和向上取整 `((count + chunk_size - 1) / chunk_size)` 均匀分配块到各任务

#### 5.3 类型特化拷贝
- **分支优化**: 根据 `unit_size` 参数使用 `switch` 语句特化处理常见数据类型：
  - `case 4`: 使用 4 字节对齐拷贝，直接映射为 `float*` 指针赋值，利用 MLU 的对齐访存指令
  - `case 2`: 使用 2 字节对齐拷贝，映射为 `int16_t*` 指针赋值
  - `default`: 回退到通用 `__memcpy` 函数（GDRAM2GDRAM 拷贝模式）
- **性能收益**: 特化拷贝避免了通用内存拷贝函数的额外开销，提升常见数据类型的拷贝吞吐量

#### 5.4 2D 拷贝加速（针对纯转置）
- **自动检测**: 在 `create` 函数中检测特定步幅模式：
  ```cpp
  // 检测条件：2 维张量且满足
  // src_strides[0] == 1 && dst_strides[1] == 1 (最内层连续)
  // src_strides[1] == y_shape[0] && dst_strides[0] == y_shape[1] (纯转置)
  ```
- **优化核函数**: `rearrange2d` 使用 MLU 的 3D 内存拷贝指令 `__memcpy` 的一次调用完成多行多列的批量拷贝，比逐元素拷贝效率高数倍
- **参数映射**:
  - `__memcpy` 的 12 参数接口支持 3D 数组拷贝，指定源和目标的行跨距（stride）、列跨距、元素大小和行列数量
  - 参数布局：`(dst_base, src_base, element_size, GDRAM2GDRAM, dst_stride_row, rows-1, dst_stride_col, cols-1, src_stride_row, rows-1, src_stride_col, cols-1)`
- **列分块策略**: 每个任务处理 16 列（`cols_per_task = 16`），减少每个任务的拷贝次数，提高指令级并行

#### 5.5 算法复杂度
- **通用 rearrange 核函数**:
  - 时间复杂度：O(count × ndim)，每个元素需要 ndim 次除法和取模计算坐标偏移
  - 空间复杂度：O(ndim) 设备内存（步幅数组），O(1) 寄存器（局部变量）
- **优化 rearrange2d 核函数**:
  - 时间复杂度：O(count)，通过硬件加速的 3D 拷贝指令实现，接近内存带宽极限
  - 空间复杂度：O(1)，无需额外的步幅数组

### 错误处理机制
- **参数验证**: 在 `create` 函数中使用 `CHECK_OR_RETURN` 宏验证：
  - 源张量和目标张量数据类型一致（`INFINI_STATUS_BAD_TENSOR_DTYPE`）
  - 维度数相等（`INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - 形状完全相同（`CHECK_SAME_SHAPE` 宏）
- **设备内存分配检查**: 每次 `cnrtMalloc` 调用后检查返回值，失败时返回 `INFINI_STATUS_INTERNAL_ERROR`
- **元数据创建检查**: 使用 `CHECK_RESULT` 宏检查 `RearrangeMeta::create` 的返回状态，传播错误码
- **队列同步**: 在 `calculate` 函数末尾调用 `cnrtQueueSync(queue)` 确保核函数执行完成，捕获执行错误

### 依赖关系
- **外部依赖**:
  - `"../../../devices/bang/bang_handle.h"`: 提供 `device::bang::Handle` 类，封装 MLU 设备和上下文
  - `"../../../devices/bang/common_bang.h"`: 提供 BANG 语言通用定义和工具函数
  - `"../rearrange.h"`: 提供 `DESCRIPTOR` 宏定义，自动生成 `Descriptor` 类框架
- **工具函数**: `utils::RearrangeMeta` 工具类负责计算重排操作的索引映射和步幅信息（定义于 `/home/qy/src/Infini/InfiniCore/src/utils/rearrange.h`）
- **硬件抽象层**:
  - CNRT (Cambricon Runtime) API: `cnrtMalloc`, `cnrtFree`, `cnrtMemcpyAsync`, `cnrtQueueCreate`, `cnrtQueueSync`, `cnrtQueueDestroy`
  - BANG 语言扩展: `__mlu_global__` (核函数标记), `taskId` (任务 ID 内置变量), `taskDimX`, `taskDimY` (任务维度内置变量), `__memcpy` (优化内存拷贝)

### 设计模式
- **RAII (Resource Acquisition Is Initialization)**: `Descriptor` 类在构造时分配所有资源（设备内存、元数据），在析构时自动释放，防止资源泄漏
- **Strategy Pattern (策略模式)**: 通过 `use_2d_copy` 标志在运行时选择不同的核函数实现（通用 rearrange 或优化 rearrange2d），根据输入特征自动选择最优策略
- **Opaque Pointer Pattern (不透明指针模式)**: 将实现细节封装在 `Opaque` 结构体中，对外部隐藏设备侧资源和内部状态，提高 ABI 稳定性
- **Template Method Pattern (模板方法模式)**: `DESCRIPTOR` 宏定义了算子的标准生命周期框架（create → calculate → destroy），各硬件后端（bang, cuda, cpu）继承并实现具体逻辑

### 硬件特性适配
- **MLU 架构特点**:
  - 采用集群 (Cluster) 并行架构，本实现使用 4 集群 × 10 任务单元 = 40 并行任务
  - GDRAM (Global DRAM) 带宽高但延迟大，因此采用分块 (chunk) 和寄存器预取策略掩盖延迟
  - 支持 3D 内存拷贝指令 `__memcpy`，可一次性完成多维数组的行/列批量拷贝
- **内存对齐**: 所有数据访问都基于 `unit_size` 进行对齐，利用 MLU 的对齐访存优化
- **任务映射**: `taskId`, `taskDimX`, `taskDimY` 是 BANG 语言的内置变量，自动映射到硬件并行单元，无需手动管理线程 ID

### 数据流示例（2D 转置）
假设输入矩阵形状 [2, 3] (2 行 3 列)，行主序存储：

```
源张量 x:
  地址布局: [a00, a01, a02, a10, a11, a12]
  物理内存: [0x1000, 0x1004, 0x1008, 0x100C, 0x1010, 0x1014] (float, 每个 4 字节)
  步幅: src_strides = [1, 2] (最内层连续)

目标张量 y (转置后 3 行 2 列):
  逻辑布局: [[a00, a10], [a01, a11], [a02, a12]]
  物理内存: [0x2000, 0x2004, 0x2008, 0x200C, 0x2010, 0x2014]
  步幅: dst_strides = [1, 3] (最内层连续)

rearrange2d 核函数参数:
  outer_dim = 2 (行数)
  inner_dim = 3 (列数)
  dst_stride_bytes = 12 (每行 3 个元素 × 4 字节)
  src_stride_bytes = 8 (每行 2 个元素 × 4 字节，转置前)
  unit_size = 4

执行流程:
  1. 任务 0 处理列块 [0-2] (全部列)
  2. 调用 __memcpy(dst + 0*4, src + 0*4, 4, GDRAM2GDRAM, 4, 1, 12, 2, 8, 1, 4, 2)
     含义：从源地址 0x1000 拷贝到目标地址 0x2000，元素大小 4 字节，
           源行跨距 8 字节，目标行跨距 12 字节，拷贝 2 行 3 列
  3. 硬件自动处理转置映射，结果写入目标地址
```

### 性能调优建议
- **Chunk 大小**: 当前设置为 256，可根据 MLU 型号的缓存大小调整（更大的缓存可使用更大的 chunk）
- **列块大小**: `cols_per_task = 16`，可根据矩阵宽度调整，宽矩阵可增大该值以减少任务数
- **集群配置**: `dim.x=4, dim.y=10` 适用于典型 MLU 设备，可根据实际硬件的集群数和计算单元数调整
- **阈值优化**: 当前仅对 2D 纯转置启用优化核函数，可扩展检测更多常见模式（如切片、广播等）
