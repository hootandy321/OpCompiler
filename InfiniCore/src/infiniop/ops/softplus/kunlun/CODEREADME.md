# Softplus Kunlun 算子核心实现文档

本模块实现了昆仑（KUNLUN）XPU 设备上的 Softplus 激活函数算子，基于通用逐元素（elementwise）运算框架构建，支持 FP16、BF16 和 FP32 三种数据类型，采用模板元编程实现类型安全的设备端计算。

## 1. 模块结构

- **`kernel.h`**: 定义 Softplus 运算符的设备端核心实现，包含 CUDA/XPU 兼容的设备函数模板，处理不同精度的类型转换和数值计算
- **`softplus_kunlun.h`**: 算子描述符的公共头文件，通过宏复用通用逐元素运算接口
- **`softplus_kunlun.xpu`**: 算子描述符的具体实现，包含算子创建（create）和计算调度（calculate）逻辑，继承自通用逐元素基类

## 2. 核心类与结构体

### `SoftplusOp`
- **位置**: `kernel.h`
- **主要功能**: 设备端软加（softplus）激活函数的函数对象实现，对输入张量逐元素应用 softplus 变换
- **关键成员**:
  - `num_inputs`: 静态常量，值为 1，表示该运算符接受 1 个输入张量
- **核心方法**:
  - `operator()(const T *inputs) const`: 重载函数调用运算符，执行 softplus 计算
    - **算法逻辑**: softplus(x) = log(1 + exp(x))，当 x > 20 时直接返回 x 以避免数值溢出
    - **类型特化**:
      - `half` (FP16): 通过 `__half2float` 转为 float 计算，结果用 `__float2half` 转回
      - `bfloat16_t` (BF16): 通过 `__bfloat162float` 转为 float 计算，结果用 `__float2bfloat16` 转回
      - `float` (FP32): 直接在 float 精度下计算
    - **时间复杂度**: O(1) 每元素，包含对数和指数运算
    - **数值稳定性**: 当输入值 > 20 时直接返回输入值，避免 `exp(x)` 溢出
- **生命周期**: 编期期静态常量对象，无运行时构造/析构开销

### `Descriptor` 类
- **位置**: `softplus_kunlun.xpu`
- **命名空间**: `op::softplus::kunlun`
- **主要功能**: 算子描述符，封装算子元数据和设备实现，继承自通用逐元素描述符基类
- **关键成员** (继承自基类):
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型（F16/BF16/F32）
  - `_info`: `op::elementwise::ElementwiseInfo`，包含张量形状、步长、广播、连续性等元数据
  - `_device_info`: `op::elementwise::kunlun::DeviceImpl *`，昆仑设备实现指针，用于启动内核
  - `_workspace_size`: `size_t`，所需工作空间大小（存储元数据和输入指针数组）
  - `_device`: 设备类型标识符
  - `_device_id`: 设备 ID
- **核心方法**:
  - `create(...)`: 静态工厂方法，创建并初始化算子描述符
    - **参数验证**:
      - 检查数据类型是否为 F16/BF16/F32（`CHECK_DTYPE` 宏）
      - 验证输入和输出张量形状完全一致（`CHECK_SAME_SHAPE` 宏）
    - **元数据构建**: 调用 `op::elementwise::ElementwiseInfo::create()` 生成逐元素运算元数据
    - **工作空间计算**: `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`，包含元数据和输入指针数组
    - **设备实现创建**: 调用 `op::elementwise::kunlun::DeviceImpl::create()` 初始化昆仑设备实现
    - **宏展开**: `CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR` 宏自动构造 Descriptor 对象并赋值成员变量
  - `calculate(...)`: 执行软加计算
    - **工作空间检查**: 验证传入 workspace 大小是否满足 `_workspace_size` 要求
    - **类型分发**: 根据 `_dtype` 调用对应的模板特化：
      - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<8, kunlun::SoftplusOp, half>(...)`
      - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<8, kunlun::SoftplusOp, bfloat16_t>(...)`
      - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<8, kunlun::SoftplusOp, float>(...)`
    - **模板参数**: `8` 表示内核块大小，`kunlun::SoftplusOp` 为运算符，第三参数为数据类型
  - `~Descriptor()`: 默认析构函数，自动释放基类资源（智能指针管理）

## 3. API 接口

```cpp
// 算子创建接口（通过宏展开的静态方法）
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // 昆仑设备句柄
    Descriptor **desc_ptr,                 // 输出参数：返回创建的描述符指针
    infiniopTensorDescriptor_t out_desc,   // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（单元素）
);
// 返回值：INFINI_STATUS_SUCCESS 成功；错误码包括类型错误、形状不匹配等

// 算子计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // 设备工作空间指针
    size_t workspace_size,                 // 工作空间大小
    void *output,                          // 输出张量设备指针
    std::vector<const void *> inputs,      // 输入张量设备指针向量（单元素）
    void *stream                           // 昆仑流指针
) const;
// 返回值：INFINI_STATUS_SUCCESS 成功；INFINI_STATUS_INSUFFICIENT_WORKSPACE 工作空间不足

// 设备端内核函数（通过 elementwise 框架自动调用）
template <typename T>
__device__ T SoftplusOp::operator()(const T *inputs) const;
// 设备函数，由逐元素内核框架对每个元素调用
```

## 4. 使用示例

```cpp
// 示例：在昆仑 XPU 上执行 Softplus 激活操作

// 1. 准备张量描述符（假设已有 handle）
infiniDtype_t dtype = INFINI_DTYPE_F16;
std::vector<int64_t> shape = {1024, 1024};

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc);
infiniopSetTensorDescriptor(x_desc, dtype, 2, shape.data(), nullptr);
infiniopCreateTensorDescriptor(handle, &y_desc);
infiniopSetTensorDescriptor(y_desc, dtype, 2, shape.data(), nullptr);

// 2. 分配设备内存
void *d_x, *d_y;
size_t nbytes = 1024 * 1024 * sizeof(half);
xpu_malloc(&d_x, nbytes);
xpu_malloc(&d_y, nbytes);

// 3. 创建算子描述符
std::vector<infiniopTensorDescriptor_t> input_descs = {x_desc};
op::softplus::kunlun::Descriptor *softplus_desc = nullptr;
auto status = op::softplus::kunlun::Descriptor::create(
    handle, &softplus_desc, y_desc, input_descs);

// 4. 分配工作空间（基于查询的大小）
void *workspace = nullptr;
xpu_malloc(&workspace, softplus_desc->_workspace_size);

// 5. 执行计算
kunlunStream_t stream;
xpu_stream_create(&stream);
std::vector<const void *> inputs = {d_x};
status = softplus_desc->calculate(workspace, softplus_desc->_workspace_size,
                                 d_y, inputs, stream);

// 6. 同步并清理
xpu_stream_synchronize(stream);
delete softplus_desc;
xpu_free(workspace);
xpu_free(d_x);
xpu_free(d_y);
```

## 5. 实现细节

### 算法与数值稳定性
- **Softplus 公式**: `softplus(x) = log(1 + exp(x))`，是 ReLU 的平滑近似
- **溢出优化**: 当 `x > 20.0f` 时直接返回 `x`，因为 `exp(20) ≈ 4.85e8`，此时 `log(1 + exp(x)) ≈ x`，避免了计算 `exp(x)` 时的浮点溢出
- **类型转换策略**: FP16/BF16 在设备端转换为 FP32 进行计算，保证数值精度，计算完成后转回原类型

### 内存管理与工作空间
- **工作空间组成**:
  - 元数据区域（`info.getMetaMemSize()`）：存储 ndim、shape、strides、contiguous/broadcast 标志
  - 输入指针数组（`info.getInputSize() * sizeof(void*)`）：存储所有输入张量的设备指针
- **设备内存传输**: 通过 `xpu_memcpy_async` 异步将主机端元数据和指针数组复制到设备工作空间
- **指针布局**: 在设备端通过指针偏量化访问工作空间中的不同元数据段，避免多次分配

### 并发与内核调度
- **内核启动配置**: `<<<BLOCK_SIZE, 64, stream>>>`，其中 BLOCK_SIZE=8（在 calculate 中硬编码），每块 64 个线程
- **并行策略**:
  - 二维线程索引：`thread_id = ncores * cluster_id() + cid`
  - 总线程数：`nthreads = ncores * cluster_num()`
  - 循环分块：每次处理 `min(BUFF_SIZE, roundup_div(output_size, nthreads))` 个元素，BUFF_SIZE=64
- **本地内存优化**:
  - 使用 `__local__` 本地内存缓存输入数据（`inputs_buf[N]`），减少全局内存访问
  - 本地内存存储元数据（shape、strides、标志），所有核心共享加载
- **内存屏障**: 在 GM2LM/LM2GM 传输后调用 `mfence()`，在循环结束后调用 `sync_cluster()` 保证一致性

### 广播与非连续张量支持
- **InputIndexer**: 根据输入张量的步长（strides）和广播标志，将线性索引转换为实际内存偏移
  - 连续张量：直接使用线性索引
  - 非连续张量：调用 `indexToOffset()` 根据形状和步长计算偏移量
- **OutputIndexer**: 类似输入索引器，处理输出张量的非连续布局

### 错误处理
- **类型检查**: 编译期 `if constexpr` + 运行时 `CHECK_DTYPE` 宏双重保证
- **形状验证**: `CHECK_SAME_SHAPE` 宏确保输入输出形状一致（softplus 不支持广播改变形状）
- **工作空间验证**: calculate 方法检查传入 workspace_size 是否满足要求
- **状态码**: 使用 `infiniStatus_t` 枚举返回详细错误信息（成功、类型错误、工作空间不足等）

### 设计模式
- **CRTP（奇异递归模板模式）**: `SoftplusOp` 作为策略类，通过模板参数传入通用逐元素内核框架
- **策略模式**: `SoftplusOp` 封装具体运算逻辑，与调度框架解耦
- **工厂模式**: `create()` 静态方法封装复杂构造逻辑
- **模板方法模式**: 基类定义算法骨架（元数据处理、内核启动），子类提供运算符实现
- **RAII**: 使用 `std::shared_ptr` 管理设备实现资源，自动释放

### 依赖关系
- **内部依赖**:
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun.h`: 通用逐元素运算实现
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun_api.h`: 设备接口和宏定义
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`: 通用逐元素元数据类
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/kunlun/`: 昆仑设备句柄和内核工具
- **外部依赖**:
  - 昆仑 XPU SDK（`xpu_memcpy_async`, `xpu_malloc`, `__global_ptr__`, `__local__` 等）
  - C++17 标准库（`std::is_same_v`, `std::vector`, `std::shared_ptr`）

### 性能特征
- **计算复杂度**: O(n)，其中 n 为张量元素总数，每元素包含 exp、log、加法运算
- **内存带宽**: 读取 1 个输入，写入 1 个输出，共 2 * sizeof(T) * n 字节传输
- **并行度**: 理论上可扩展到所有昆仑计算核心，实际受张量大小和内存带宽限制
- **优化技术**:
  - 循环分块（BUFF_SIZE=64）提高缓存利用率
  - 本地内存缓存减少全局内存延迟
  - 异步内存传输与计算重叠
  - 编译期模板特化消除类型分支开销
