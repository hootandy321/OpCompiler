# 目录: tensor 架构全景

## 1. 子系统职责

`tensor` 目录是 InfiniCore 的**核心数据结构层**，负责实现张量（Tensor）这一抽象概念。张量是深度学习框架中最基础的数据容器，所有算子的输入输出都表现为张量。该子系统不包含子目录，是一个纯粹的叶节点模块，通过四个 C++ 源文件提供完整的张量管理能力。

**核心价值**：
- 提供统一的张量抽象，封装多维数组、内存管理、形状变换、跨设备数据传输等功能
- 支持零拷贝视图变换（view），通过步长（stride）机制实现高效的切片、转置、重塑等操作
- 实现跨设备内存拷贝（CPU ↔ GPU 等），为异构计算提供数据传输基础设施
- 集成 InfiniOP 张量描述符（infiniopTensorDescriptor_t），实现与底层算子库的无缝对接
- 提供调试工具，支持张量内容的打印与二进制导出

**设计模式**：
- **PImpl 模式**：Tensor 作为句柄类，TensorImpl 作为实现类，通过 std::shared_ptr 管理生命周期
- **RAII 资源管理**：Memory 对象自动管理设备内存的分配与释放
- **视图共享机制**：通过共享底层 Memory 对象 + 不同 offset/shape/strides 实现零拷贝视图

---

## 2. 模块导航

* **tensor.cc**（张量核心实现）：
    * *功能*：实现张量的创建、销毁、元信息查询等核心功能
    * *职责*：提供张量构造函数、内存分配（empty/zeros/ones）、外部内存封装（from_blob）、形状与步长管理
    * *关键类*：TensorImpl（实现类）、TensorMetaData（元数据封装）、TensorData（数据指针封装）

* **copy.cc**（跨设备数据传输）：
    * *功能*：实现张量在不同设备间的数据拷贝与连续化操作
    * *职责*：处理 CPU ↔ GPU、GPU ↔ GPU 等跨设备传输，自动检测源/目标张量的连续性并选择最优拷贝策略
    * *关键方法*：to(device)、copy_from(src)、contiguous()

* **view.cc**（视图变换）：
    * *功能*：实现零拷贝的张量形状变换操作
    * *职责*：提供维度增删（squeeze/unsqueeze）、切片（narrow）、维度重排（permute）、重塑（view/as_strided）等视图操作
    * *设计特点*：所有视图操作共享底层内存，仅通过修改 shape/strides/offset 实现不同视角

* **debug.cc**（调试与导出）：
    * *功能*：提供张量内容的打印与二进制文件导出功能
    * *职责*：支持所有数据类型（F16/BF16/F32/I32 等）的可视化输出，自动处理设备同步与类型转换
    * *关键方法*：debug()、debug(filename)

---

## 3. 架构逻辑图解

### 3.1 核心数据结构

```
┌────────────────────────────────────────────────────────────┐
│                        Tensor (句柄)                        │
│  - std::shared_ptr<TensorImpl> impl_                        │
│  - operator->() 转发到 TensorImpl                            │
└────────────────────────────────────────────────────────────┘
                            ↓ 智能指针管理
┌────────────────────────────────────────────────────────────┐
│                     TensorImpl (实现)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TensorMetaData meta_ (元信息)                        │  │
│  │    - Shape shape (形状，如 [2, 3, 4])                  │  │
│  │    - Strides strides (步长，如 [12, 4, 1])             │  │
│  │    - DataType dtype (数据类型，F16/F32/I32 等)         │  │
│  │    - infiniopTensorDescriptor_t desc (InfiniOP 描述符) │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TensorData data_ (数据指针)                          │  │
│  │    - size_t offset (偏移量，字节)                      │  │
│  │    - std::shared_ptr<Memory> memory (底层内存对象)     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            ↓ 引用
┌────────────────────────────────────────────────────────────┐
│                      Memory (内存管理)                      │
│  - std::byte* data_ (原始指针)                              │
│  - size_t size_ (总字节数)                                  │
│  - Device device_ (所属设备)                                │
│  - Deleter deleter_ (自定义删除器，支持 CUDA free 等)       │
└────────────────────────────────────────────────────────────┘
```

### 3.2 张量创建流程

#### 场景 1：创建空张量（Tensor::empty）

```
1. 用户调用
   Tensor::empty({2, 3}, DataType::F32, Device::cuda(0))

2. 计算连续步长
   calculate_contiguous_strides({2, 3}) → [3, 1]

3. 创建 TensorImpl
   new TensorImpl(shape={2, 3}, strides=[3, 1], dtype=F32)

4. 分配设备内存
   if (device == CPU):
       context::allocateHostMemory(2 * 3 * sizeof(float))
   else:
       context::setDevice(cuda(0))
       context::allocateMemory(2 * 3 * sizeof(float))
   → 返回 std::shared_ptr<Memory>

5. 封装为 Tensor
   return Tensor{tensor_impl}
```

#### 场景 2：从外部内存创建（Tensor::from_blob）

```
1. 外部准备内存
   float* external_ptr = new float[6];  // C++ 数组
   或者 cudaMalloc(&gpu_ptr, size);      // CUDA 内存

2. 创建张量（不拷贝数据，仅封装）
   Tensor::from_blob(external_ptr, {2, 3}, DataType::F32, Device::cpu())

3. 创建 Memory 对象（自定义删除器为 nullptr）
   Memory(external_ptr, 6 * sizeof(float), Device::cpu(), nullptr)

4. 返回 Tensor
   → 用户负责外部内存的生命周期管理
```

### 3.3 视图变换机制

视图变换的核心思想：**共享底层内存，通过不同的 shape/strides/offset 访问同一数据**。

#### 操作 1：重塑（view）

```
原始张量: shape=[2, 3], strides=[3, 1], 数据=[0,1,2,3,4,5]
         内存布局: [0][1][2][3][4][5]

执行: tensor.view({3, 2})

步长计算逻辑:
1. 检查总元素数: 2*3 = 3*2 = 6 ✓
2. 查看原始步长: [3, 1]
   - 维度 0 (size=2, stride=3) 和维度 1 (size=3, stride=1) 是连续的
   - 合并为虚拟维度 (size=6, stride=1)
3. 拆分虚拟维度以匹配新形状 [3, 2]:
   - 新维度 0 (size=3): stride = 1 * (6/3) = 2
   - 新维度 1 (size=2): stride = 2 * (2/2) = 1

结果张量: shape=[3, 2], strides=[2, 1], 共享同一内存
访问 view[1, 0] → 原始索引 1*2 + 0*1 = 2 → 数据值 2
```

#### 操作 2：转置（permute）

```
原始张量: shape=[2, 3, 4], strides=[12, 4, 1]

执行: tensor.permute({2, 0, 1})  // 将维度顺序从 [0,1,2] 变为 [2,0,1]

新形状计算:
- new_shape[0] = old_shape[2] = 4
- new_shape[1] = old_shape[0] = 2
- new_shape[2] = old_shape[1] = 3
→ new_shape = [4, 2, 3]

新步长计算:
- new_strides[0] = old_strides[2] = 1
- new_strides[1] = old_strides[0] = 12
- new_strides[2] = old_strides[1] = 4
→ new_strides = [1, 12, 4]

结果: 零拷贝转置，数据仍为原始内存布局，通过步长实现访问
```

#### 操作 3：切片（narrow）

```
原始张量: shape=[3, 4], strides=[4, 1]
         数据: [[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9,10,11]]

执行: tensor.narrow({{0, 1, 2}, {1, 2, 2}})
       - 在维度 0 上从索引 1 开始，取长度 2（即行 1-2）
       - 在维度 1 上从索引 2 开始，取长度 2（即列 2-3）

新形状: [2, 2]

偏移量计算:
offset = 1 * strides[0] * sizeof(float)  // 跳过 1 行
       + 2 * strides[1] * sizeof(float)  // 跳过 2 列
       = 1 * 4 * 4 + 2 * 1 * 4 = 24 字节

结果: 指向内存偏移 24 字节处的视图
     数据视图: [[6, 7],
                [10,11]]
```

### 3.4 跨设备数据传输流程

copy.cc 实现了智能的跨设备拷贝，根据源/目标设备的连续性选择最优策略。

#### 场景 1：GPU → CPU 连续张量拷贝

```
1. 用户调用
   cpu_tensor = gpu_tensor.to(Device::cpu())

2. 检测目标设备不同
   if (src.device != dst.device): 进入跨设备路径

3. 检查源张量连续性
   if (!src->is_contiguous()):
       src = src->contiguous()  // 先连续化（调用 rearrange 算子）

4. 执行 D2H 拷贝
   context::setDevice(src.device)  // 设置 CUDA 上下文
   context::memcpyD2H(
       dst->data(),    // CPU 指针
       src->data(),    // GPU 指针
       copy_size       // nbytes() 字节数
   )

5. 返回 CPU 张量
```

#### 场景 2：GPU → CPU 非连续张量拷贝

```
1. 用户调用
   cpu_tensor = gpu_tensor.permute({1, 0}).to(Device::cpu())

2. 检测源张量非连续
   if (!src->is_contiguous()):

3. 分配临时 GPU 连续缓冲区
   temp_gpu = Tensor::empty(src.shape(), src.dtype., src.device())

4. 拷贝 GPU → GPU（连续化）
   context::memcpyD2D(
       temp_gpu->data(),
       src->data(),
       src->data().memory->size()  // 拷贝整个内存块
   )

5. 在 GPU 上执行 rearrange 算子
   op::rearrange_(temp_gpu, src)  // 重排数据为连续布局

6. 执行 D2H 拷贝
   context::memcpyD2H(cpu_tensor->data(), temp_gpu->data(), nbytes)

7. 最终 CPU 端 rearrange（如果目标张量也非连续）
   if (!cpu_tensor->is_contiguous()):
       op::rearrange_(cpu_tensor, temp_gpu_cpu)
```

### 3.5 连续性判断与优化

连续张量的定义：内存布局与行优先（C-style）多维数组一致，即步长满足 `stride[i] = stride[i+1] * shape[i+1]`。

```cpp
bool TensorImpl::is_contiguous() const {
    Stride expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            return false;  // 步长不匹配，非连续
        }
        expected_stride *= shape[i];
    }
    return true;  // 所有维度步长都符合连续规则
}
```

**优化意义**：
- 连续张量可以一次性拷贝整个内存块（memcpyD2H/memcpyH2D）
- 非连续张量需要先调用 rearrange 算子重排数据，增加额外开销
- 算子库通常对连续张量有更优的 kernel 实现

### 3.6 调试与导出机制

debug.cc 提供了张量内容的可视化与导出功能，自动处理以下复杂情况：

#### 功能 1：控制台打印（debug()）

```
1. 同步设备
   context::syncDevice()  // 确保 GPU 计算完成

2. 拷贝到 CPU
   cpu_tensor = this->contiguous()->to(Device::cpu())

3. 根据数据类型递归打印
   - F16/BF16: 转换为 float32 后打印（避免乱码）
   - F32/F64: 直接打印
   - I32/I64: 打印整数
   - BOOL: 打印 true/false

4. 递归遍历维度
   print_data(data, shape, strides, dim=0):
       if (dim == last_dim):
           打印当前行
       else:
           for i in shape[dim]:
               递归 print_data(data + i*strides[dim], ...)
```

#### 功能 2：二进制导出（debug(filename)）

```
1. 同步设备并拷贝到 CPU
   （同上）

2. 打开二进制文件
   std::ofstream out(filename, std::ios::binary)

3. 快速路径：连续张量
   if (cpu_tensor->is_contiguous()):
       out.write(cpu_data, nbytes)  // 一次性写入

4. 慢速路径：非连续张量
   write_binary_data(out, data, shape, strides, dim=0):
       if (dim == last_dim):
           for i in shape[dim]:
               out.write(&data[i*strides[dim]], sizeof(T))
       else:
           递归处理高维
```

### 3.7 与 InfiniOP 的集成

TensorMetaData 维护了 `infiniopTensorDescriptor_t`，这是与 InfiniOP 算子库的桥梁：

```cpp
TensorMetaData::TensorMetaData(...) {
    infiniopCreateTensorDescriptor(
        &desc,
        shape.size(),           // ndim
        shape.data(),           // shape 数组
        strides.data(),         // strides 数组
        (infiniDtype_t)dtype    // 数据类型枚举
    );
}
```

**工作流程**：
1. 创建张量时同步创建 InfiniOP 描述符
2. 调用算子时传入 `tensor->desc()`，InfiniOP 根据描述符解析形状与步长
3. 张量销毁时自动销毁描述符

---

## 4. 关键技术特性

### 4.1 智能步长计算
- **连续张量**：自动计算行优先步长（如 shape=[2,3] → strides=[3,1]）
- **非连续张量**：支持任意步长组合，实现转置、广播等高级视图
- **view 操作的步长推断**：通过维度合并/拆分算法，自动计算兼容新形状的步长

### 4.2 零拷贝视图优化
- 所有视图操作（squeeze/unsqueeze/narrow/permute/view/as_strided）共享底层内存
- 仅修改 shape/strides/offset，无需数据拷贝
- 通过 std::shared_ptr 的引用计数自动管理内存生命周期

### 4.3 跨设备智能传输
- 自动检测张量连续性，非连续张量先连续化再拷贝
- 支持 CPU ↔ GPU、GPU ↔ GPU 等多种传输路径
- 利用 pinned memory 优化 PCIe 传输带宽

### 4.4 类型安全的调试工具
- 支持 12 种数据类型（F16/BF16/F32/F64/I8/I16/I32/I64/U8/U16/U32/U64/BOOL）
- F16/BF16 自动转换为 float32 打印，避免乱码
- 非连续张量通过步长正确遍历，保证输出内容与逻辑视图一致

### 4.5 InfiniOP 描述符同步
- 张量元数据变更时自动更新 InfiniOP 描述符
- 确保算子库始终获得正确的形状与步长信息
- 析构时自动释放描述符资源

---

## 5. 设计权衡与限制

### 5.1 已知限制
1. **zeros/ones 未实现**：当前仅调用 empty()，未填充实际数据（待完善）
2. **视图安全检查不足**：as_strided() 标记为 "insecure"，允许创建越界视图
3. **错误处理不完整**：部分异常情况仅抛出 generic std::runtime_error

### 5.2 性能考虑
1. **连续性检查开销**：is_contiguous() 每次遍历所有维度，可缓存结果优化
2. **视图操作的步长推断**：view() 的维度合并/拆分算法复杂度为 O(n)
3. **跨设备拷贝的临时缓冲区**：非连续张量传输需要额外 GPU 内存

### 5.3 设计优势
1. **PImpl 模式**：Tensor 句柄类支持值语义（拷贝、移动、函数传参）
2. **引用计数内存管理**：多个视图共享同一内存，自动释放最后引用
3. **与 InfiniOP 深度集成**：避免描述符转换开销，提供原生算子性能

---

## 6. 与其他模块的交互

### 6.1 依赖 context 模块
- `context::allocateMemory()`：分配设备内存
- `context::setDevice()`：设置当前设备上下文
- `context::memcpyD2H/H2D/D2D()`：跨设备数据传输
- `context::syncDevice()`：同步设备操作

### 6.2 依赖 ops 模块
- `op::rearrange_()`：连续化非连续张量
- `op::rearrange()`：返回连续化的新张量

### 6.3 被以下模块依赖
- **所有算子**（ops/）：算子接受 Tensor 作为输入/输出
- **神经网络模块**（nn/）：Linear、Embedding 等层内部维护参数张量
- **Python 绑定**（pybind11/）：将 TensorImpl 暴露为 Python 的 tensor.Tensor
- **计算图**（graph/）：GraphTensor 包装 TensorImpl，支持图模式

---

## 7. 使用示例（C++ API）

```cpp
using namespace infinicore;

// 1. 创建张量
auto tensor = Tensor::empty({2, 3}, DataType::F32, Device::cuda(0));

// 2. 视图变换
auto view = tensor.view({3, 2});              // 重塑
auto transposed = tensor.permute({1, 0});     // 转置
auto sliced = tensor.narrow({{0, 1, 1}});     // 切片第 0 维

// 3. 跨设备传输
auto cpu_tensor = tensor.to(Device::cpu());

// 4. 连续化
auto contiguous = transposed.contiguous();

// 5. 调试
tensor.debug();              // 打印到控制台
tensor.debug("data.bin");    // 导出到二进制文件

// 6. 元信息查询
std::cout << tensor->shape() << std::endl;        // [2, 3]
std::cout << tensor->strides() << std::endl;      // [3, 1]
std::cout << tensor->is_contiguous() << std::endl; // true
```

---

## 8. 总结

`tensor` 目录实现了 InfiniCore 框架的**数据基础设施**，通过精心设计的 PImpl 模式、视图共享机制、智能跨设备传输，提供了高性能、易用性、类型安全的张量抽象。该模块是整个框架的基石，所有上层功能（算子、神经网络、计算图）都建立在张量这一统一数据结构之上。

**核心亮点**：
1. 零拷贝视图变换，避免不必要的数据复制
2. 自动优化的跨设备传输，根据连续性选择最优路径
3. 与 InfiniOP 深度集成，提供原生算子性能
4. 完善的调试工具，支持所有数据类型的可视化与导出
