# Tensor Core Implementation Documentation

InfiniLM 的张量核心实现模块，提供多设备支持的张量存储、描述、变换和内存管理功能。该模块是深度学习推理框架的基础数据结构，支持跨设备（CPU、CUDA等）的张量操作，并集成了 InfiniRT 和 InfiniOp 后端。

## 1. Module Structure

- **`strorage.cpp`**: 底层内存存储管理，实现同步/异步内存分配、内存池管理和设备内存生命周期
- **`tensor.cpp`**: 核心张量类实现，提供张量创建、视图变换、数据加载、调试打印等核心功能
- **`transform.cpp`**: 张量变换操作，包括切片、维度合并/拆分、置换等形状变换操作

## 2. Core Classes

### `Storage`
- **Location**: `strorage.cpp`, `tensor.hpp`
- **Primary Function**: 封装底层设备内存（GPU/CPU）的分配、释放和管理，支持内存池复用和异步分配
- **Key Members**:
  - `void *_memory`: 原始设备内存指针，通过 InfiniRT 分配
  - `size_t _size`: 内存块字节数大小
  - `infiniDevice_t _device_type`: 设备类型（CPU/CUDA/Ascend等）
  - `int _device_id`: 设备编号，用于多卡场景
  - `std::shared_ptr<MemoryPool> _memory_pool`: 可选的内存池，用于实现高效内存复用
- **Core Methods**:
  - `create(size_t size)`: 同步分配设备内存，通过 `infinirtMalloc` 实现，自动获取当前设备上下文
  - `createAsync(size_t size, infinirtStream_t stream)`: 异步流分配，通过 `infinirtMallocAsync` 实现，用于并发优化
  - `createFromPool(size_t size, std::shared_ptr<MemoryPool> pool)`: 从内存池分配，如果 pool 为 nullptr 则回退到直接分配
  - `createHost(size_t size)`: 分配主机（CPU）锁页内存，通过 `infinirtMallocHost` 实现零拷贝优化
  - `~Storage()`: 析构时自动释放内存，内存池分配的内存调用 `pool->release()`，否则调用对应的 `infinirtFree*`
- **Lifecycle**: 使用私有构造函数，通过静态工厂方法创建，采用 `std::shared_ptr` 管理生命周期，析构时根据设备类型和内存池状态选择正确的释放策略

### `TensorDesc`
- **Location**: `tensor.cpp`, `tensor.hpp`
- **Primary Function**: 张量的元数据描述符，包含形状、步长、数据类型，提供张量布局的不变视图，支持哈希计算和连续性检测
- **Key Members**:
  - `infiniDtype_t _dtype`: 数据类型（F16/F32/BF16/I32等）
  - `std::vector<size_t> _shape`: 各维度大小
  - `std::vector<ptrdiff_t> _strides`: 各维度步长（字节偏移量），支持非连续张量
  - `infiniopTensorDescriptor_t _desc`: 惰性创建的 InfiniOp 原生描述符，用于算子调用
  - `size_t _seed`: 基于 shape 和 strides 计算的哈希值，用于缓存键值
- **Core Methods**:
  - `create(dtype, shape)`: 自动计算行主序连续步长，从最低维开始 stride=1，逐级向上乘积
  - `createWithOrder(dtype, shape, order)`: 支持自定义内存布局（如列主序），order[i] 表示第 i 维在 shape 中的原始索引
  - `desc()`: 惰性创建并返回 InfiniOp 描述符，首次调用时通过 `infiniopCreateTensorDescriptor` 创建，后续直接缓存返回
  - `isContigous()`: O(n) 时间检查张量是否为行主序连续布局，通过重新计算理想步长并与实际步长比较
  - `computeTensorDesHash()`: 使用 `hash_combine` 算法将 shape 和 strides 混合成 64 位哈希值，常量 `0x9e3779b9` 来自黄金比例哈希
  - `dimMerge(dim_start, dim_end)`: 合并连续维度，验证被合并维度的步长连续性后，将 shape 乘积合并、stride 继承最内维
  - `dimSplit(dim, dims)`: 拆分维度，要求拆分后的维度乘积等于原维度，新步长按比例缩放原步长
  - `permute(order)`: 按指定顺序重排维度，order[i] 表示新维度 i 对应的原始维度索引
- **Lifecycle**: 通过 `create` 系列工厂方法构造，析构时调用 `infiniopDestroyTensorDescriptor` 释放原生描述符

### `Tensor`
- **Location**: `tensor.cpp`, `tensor.hpp`
- **Primary Function**: 用户可见的张量接口，组合 Storage 和 TensorDesc，支持数据操作、视图变换、跨设备拷贝和调试
- **Key Members**:
  - `std::shared_ptr<Storage> _storage`: 底层内存存储，可被多个 Tensor 共享（如 view、slice）
  - `std::shared_ptr<const TensorDesc> _desc`: 不可变的张量元数据描述符
  - `ptrdiff_t _offset`: 字节偏移量，用于 slice/view 等零拷贝操作，指向 `_storage` 内部的数据起始位置
- **Core Methods**:
  - `buffer(dtype, shape, pool)`: 创建缓冲区张量，计算总字节大小后从内存池分配，自动计算连续步长
  - `weight(data, dtype, shape)`: 从主机数据创建权重张量，分配设备内存后调用 `load()` 拷贝数据
  - `load(host_data, stream)`: 将主机数据 H2D 拷贝到设备，支持异步流拷贝；同步路径使用 `std::mutex` 保护（修复沐曦平台多线程并发死锁问题）
  - `memShare(shape, dtype_)`: 共享底层存储创建新张量，验证大小不超过原存储后，复用 `_storage` 并重置 `_offset=0`
  - `slice(dim, start, len)`: 零拷贝切片，计算新形状和字节偏移（`offset = start * strides[dim] * dsize(dtype)`），共享原存储
  - `view(new_shape)`: 复杂的重塑视图算法，先合并连续维度，再按需拆分以匹配新形状，保证总元素数不变
  - `view_as(new_shape, new_strides)`: 直接指定形状和步长的底层视图操作，用于高级场景
  - `dimMerge/dimSplit/permute`: 转发到 TensorDesc 的对应操作，创建新描述符并共享存储
  - `copyFrom(src, handle, stream)`: 通过 InfiniOp 的 `infiniopRearrange` 实现跨设备/跨布局张量拷贝，自动处理步长不一致
  - `data(offset)`: 返回指向实际数据的指针，计算公式：`(char*)_storage->memory() + _offset + offset * dsize(dtype)`
  - `debug(filename)`: 调试输出，同步设备后 D2H 拷贝到 CPU，根据 dtype 打印或导出到二进制文件，支持 F16/BF16 自动转 F32 显示
- **Lifecycle**: 通过 `std::make_shared` 创建，`enable_shared_from_this` 支持从成员函数返回 `shared_ptr`，析构时自动减少 Storage 引用计数

### `MemoryPool`
- **Location**: `allocator.hpp`（依赖模块）
- **Primary Function**: 基于 Block 分配算法的内存池，支持对齐分配、空闲块合并和多重尺寸分配
- **Key Members**:
  - `std::vector<void *> _base_regions`: 从设备分配的大块基础内存区域
  - `std::set<Block> _all_blocks`: 所有块的有序集合（按地址排序），Block 包含 {base, ptr, size, is_free}
  - `std::multimap<size_t, std::set<Block>::iterator> _free_blocks`: 从大小到块迭代器的多重映射，用于最佳适配（Best Fit）查找
  - `std::unordered_map<void *, std::set<Block>::iterator> _ptr_to_block`: 从指针到块的快速索引，O(1) 释放查找
  - `size_t _alignment`: 对齐字节数，默认 256 字节
- **Core Methods**:
  - `alloc(size)`: 最佳适配算法，在 `_free_blocks` 中查找 >= size 的最小块，如果找不到则调用 `allocateNewRegion` 扩展；分配时将块分割为已使用部分和剩余空闲部分
  - `release(ptr)`: O(1) 查找块，标记为空闲后调用 `tryCoalesce` 合并相邻空闲块
  - `tryCoalesce(block)`: 尝试与前驱和后继空闲块合并，减少碎片
- **Lifecycle**: 构造时可指定初始大小和 alignment，析构时释放所有 `_base_regions`

## 3. API Interface

```cpp
// Tensor 工厂方法
std::shared_ptr<Tensor> Tensor::buffer(infiniDtype_t dtype,
                                       const std::vector<size_t> &shape,
                                       std::shared_ptr<MemoryPool> pool = nullptr);
// 从内存池分配设备内存，创建零初始化张量，返回共享指针

std::shared_ptr<Tensor> Tensor::weight(void *host_data,
                                       infiniDtype_t dtype,
                                       const std::vector<size_t> &shape);
// 从主机数据创建权重张量，自动 H2D 拷贝，用于加载模型参数

// 数据操作
void Tensor::load(const void *host_data, infinirtStream_t stream = nullptr);
// 同步或异步地将主机数据拷贝到设备内存，同步路径使用 mutex 保护

void Tensor::copyFrom(std::shared_ptr<Tensor const> src,
                      infiniopHandle_t handle,
                      infinirtStream_t stream = nullptr);
// 通过 InfiniOp Rearrange 算子实现张量拷贝，自动处理跨设备和步长转换

// 视图变换（零拷贝）
std::shared_ptr<Tensor> Tensor::view(const std::vector<size_t> &new_shape) const;
// 重塑张量形状，要求元素总数不变，自动计算步长或使用复杂合并拆分算法

std::shared_ptr<Tensor> Tensor::slice(size_t dim, size_t start, size_t len);
// 沿指定维度切片，共享底层存储，计算新的偏移量

std::shared_ptr<Tensor> Tensor::permute(const std::vector<size_t> &order);
// 重排维度，order[i] 表示新维度 i 对应的原始维度

std::shared_ptr<Tensor> Tensor::dimMerge(size_t dim_start, size_t dim_end);
// 合并连续维度 [dim_start, dim_end]，减少维度数

std::shared_ptr<Tensor> Tensor::dimSplit(size_t dim, const std::vector<size_t> &dims);
// 拆分维度 dim 为多个子维度，要求子维度乘积等于原维度

// 属性访问
const std::vector<size_t> &Tensor::shape() const;  // 形状
const std::vector<ptrdiff_t> &Tensor::strides() const;  // 步长
size_t Tensor::ndim() const;  // 维度数
infiniDtype_t Tensor::dtype() const;  // 数据类型
infiniDevice_t Tensor::deviceType() const;  // 设备类型
int Tensor::deviceId() const;  // 设备编号
size_t Tensor::numel() const;  // 元素总数（shape 的乘积）
bool Tensor::isContigous() const;  // 是否为行主序连续张量
ptrdiff_t Tensor::dataOffset() const;  // 字节偏移量

// 调试工具
void Tensor::debug(const std::string &filename = "") const;
// 打印张量内容到 stdout 或导出到二进制文件，自动处理 F16/BF16 转换

// Storage 工厂方法
std::shared_ptr<Storage> Storage::create(size_t size);
// 同步分配设备内存

std::shared_ptr<Storage> Storage::createAsync(size_t size, infinirtStream_t stream = nullptr);
// 异步流分配，用于并发优化

std::shared_ptr<Storage> Storage::createFromPool(size_t size, std::shared_ptr<MemoryPool> pool = nullptr);
// 从内存池分配或回退到直接分配

std::shared_ptr<Storage> Storage::createHost(size_t size);
// 分配主机锁页内存，用于零拷贝优化

// TensorDesc 工厂方法
static std::shared_ptr<TensorDesc> TensorDesc::create(infiniDtype_t dtype,
                                                      const std::vector<size_t> &shape);
// 创建连续布局的张量描述符，自动计算步长

static std::shared_ptr<TensorDesc> TensorDesc::createWithOrder(infiniDtype_t dtype,
                                                                const std::vector<size_t> &shape,
                                                                const std::vector<size_t> &order);
// 创建自定义内存布局的张量描述符，支持列主序等非标准布局

// 工具函数
inline size_t dsize(infiniDtype_t dtype);
// 返回数据类型的字节数，支持 F16(2), F32(4), BF16(2), I32(4) 等

inline float f16_to_f32(uint16_t h);
// IEEE 754 半精度浮点转单精度，处理归一化、非归一化、Inf、NaN

inline float bf16_to_f32(uint16_t val);
// BFloat16 转 Float32，只需移位到高 16 位（指数和尾数相同）
```

## 4. Usage Example

```cpp
// 场景 1: 创建缓冲区张量并加载数据
auto pool = std::make_shared<MemoryPool>(1 << 30);  // 1GB 内存池
auto tensor = Tensor::buffer(INFINI_DTYPE_F32, {2, 3, 4}, pool);

float host_data[24] = {/*...初始化数据...*/};
tensor->load(host_data);  // 同步 H2D 拷贝

// 场景 2: 从权重文件创建模型参数
extern void *weight_buffer;  // 预加载的权重文件
auto weight = Tensor::weight(weight_buffer, INFINI_DTYPE_F16, {128, 768, 1024});

// 场景 3: 零拷贝切片和视图变换
auto sliced = tensor->slice(1, 1, 2);  // 沿维度 1 切片 [start=1, len=2]
// sliced 共享 tensor 的底层存储，仅调整 offset 和 shape

auto reshaped = tensor->view({6, 4});  // 2x3x4 -> 6x4 重塑
// 自动合并维度再拆分，验证 2*3*4 == 6*4

// 场景 4: 维度操作
auto merged = tensor->dimMerge(1, 2);  // 合并维度 1 和 2: 2x3x4 -> 2x12
// 验证维度连续性后乘积形状，继承最内维步长

auto split = tensor->dimSplit(1, {1, 2});  // 拆分维度 1: 2x3x4 -> 2x1x2x4
// 要求 1*2 == 3，新步长按比例缩放

auto permuted = tensor->permute({2, 0, 1});  // 维度重排: 2x3x4 -> 4x2x3
// 新维度 i 来自原始维度 order[i]

// 场景 5: 跨设备拷贝（通过 InfiniOp）
infiniopHandle_t handle;  // 假设已初始化
infinirtStream_t stream = nullptr;
auto dst = Tensor::buffer(INFINI_DTYPE_F32, {2, 3, 4});
dst->copyFrom(tensor, handle, stream);
// 自动处理步长不一致和跨设备拷贝

// 场景 6: 内存池共享
auto shared = tensor->memShare({1, 3, 4});  // 共享前 1x3x4=12 个元素
// 验证大小不超过原存储，复用 _storage，offset=0

// 场景 7: 调试输出
tensor->debug();  // 打印到 stdout
tensor->debug("dump.bin");  // 导出二进制文件

// 场景 8: 异步加载（用于并发优化）
infinirtStream_t stream;
RUN_INFINI(infinirtStreamCreate(&stream, 0));
auto async_tensor = Tensor::buffer(INFINI_DTYPE_F16, {1024, 1024});
async_tensor->load(host_data, stream);  // 异步 H2D
// 程序可以继续执行其他任务

// 场景 9: 锁页内存优化（用于频繁 H2D 传输）
auto host_storage = Storage::createHost(1024);
auto pinned_tensor = Tensor::weight(host_storage->memory(), INFINI_DTYPE_F32, {16, 16});
// 锁页内存加速 DMA 传输
```

## 5. Implementation Details

- **Memory Management**:
  - 采用三级内存管理策略：直接设备分配（Storage::create）、内存池分配（MemoryPool 的 Best Fit 算法）、主机锁页内存（Storage::createHost）
  - MemoryPool 使用 `std::set<Block>` 有序存储所有块（地址排序），`std::multimap<size_t, iterator>` 实现最佳适配查找，`std::unordered_map<void*, iterator>` 实现 O(1) 释放查找
  - 内存池分配时自动对齐到 256 字节边界，支持块分割和空闲块合并以减少碎片
  - Tensor 支持零拷贝视图（view、slice、memShare），通过共享 Storage 和调整 offset 实现

- **Concurrency**:
  - `Tensor::load()` 的同步路径使用 `static std::mutex` 保护，解决沐曦平台多线程并发 H2D 拷贝死锁问题（注释中明确记录）
  - 异步 API（createAsync、load with stream）支持并发执行，用户需自行管理流同步
  - Storage 和 Tensor 使用 `std::shared_ptr` 实现引用计数，多线程安全但数据访问需外部同步

- **Performance**:
  - TensorDesc 采用惰性初始化模式，`infiniopTensorDescriptor_t` 仅在首次调用 desc() 时创建，后续直接缓存
  - 步长计算采用动态规划：从最低维开始 stride=1，逐级向上 `stride[i] = stride[i+1] * shape[i+1]`
  - `Tensor::view()` 使用复杂合并拆分算法 O(n)：先合并连续维度（通过步长连续性检测），再按需拆分匹配新形状
  - 哈希计算使用 `hash_combine` 算法：`seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2)`，常量来自黄金比例哈希，减少冲突

- **Error Handling**:
  - 使用 `RUN_INFINI` 宏包装所有 InfiniRT/InfiniOp API 调用，失败时打印错误码、函数名、文件名、行号并 `exit(EXIT_FAILURE)`
  - 使用 `ASSERT/ASSERT_EQ` 宏进行参数校验（如形状匹配、索引边界），失败时打印详细断言信息
  - `PANIC` 宏用于不可恢复错误，直接终止程序

- **Dependencies**:
  - **InfiniRT** (`infinirt.h`): 提供跨平台内存分配（malloc/mallocAsync/mallocHost）、设备管理、内存拷贝（memcpy/memcpyAsync）和流管理
  - **InfiniOp** (`infinicore_infer.h`): 提供 `infiniopCreateTensorDescriptor` 创建算子描述符，`infiniopRearrange` 实现跨设备/跨布局张量重排
  - **MemoryPool** (allocator.hpp): 可选的内存池分配器，实现 Block 管理和空闲块合并
  - **Utils** (utils.hpp): 提供 `dsize()` 数据类型大小查询，`f16_to_f32/f32_to_f16` IEEE 754 半精度转换，`bf16_to_f32` BFloat16 转换，`hash_combine` 哈希混合

- **Design Patterns**:
  - **Factory Pattern**: Storage 和 TensorDesc 使用静态工厂方法（create/createAsync/createFromPool/createHost）封装构造逻辑
  - **RAII**: Storage 析构自动释放内存，TensorDesc 析构自动销毁 InfiniOp 描述符
  - **Shared Ownership**: Tensor 使用 `std::shared_ptr` 管理 Storage 生命周期，支持多个 Tensor 共享同一存储（view/slice）
  - **Immutable Metadata**: TensorDesc 设计为不可变元数据（除内部变换方法），通过 `std::shared_ptr<const TensorDesc>` 防止意外修改
  - **Lazy Initialization**: TensorDesc::desc() 延迟创建原生描述符，减少不必要的开销
  - **Zero-Copy Views**: view/slice/memShare 通过共享 Storage 和调整 offset 实现零拷贝变换
  - **Curiously Recurring Template Pattern (CRTP)**: Tensor 继承 `std::enable_shared_from_this`，支持成员函数返回 `std::shared_ptr<this>`
