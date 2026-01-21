# Causal Softmax Ascend Backend Implementation Documentation

华为Ascend NPU后端的因果掩码softmax算子实现，支持FP16和FP32数据类型的因果注意力掩码处理。

## 1. 模块结构

- **`causal_softmax_ascend.h`**: 声明Ascend后端Descriptor类，通过宏DESCRIPTOR定义类结构
- **`causal_softmax_ascend.cc`**: 实现基于华为CANN算子库的因果softmax算子，使用masked_fill和softmax组合实现

## 2. 核心类

### `Descriptor::Opaque`
- **位置**: `causal_softmax_ascend.cc` (第8-30行)
- **主要功能**: 封装Ascend ACL API的底层资源，包括张量描述符、设备内存分配和算子执行器
- **关键成员**:
  - `aclnnTensorDescriptor_t x`: 输入张量描述符 (batch, seq_len, total_seq_len)
  - `aclnnTensorDescriptor_t mask`: 布尔型因果掩码张量 (seq_len, total_seq_len)
  - `aclnnTensorDescriptor_t y`: 输出张量描述符
  - `aclnnTensorDescriptor_t value`: 负无穷值标量张量，用于掩码填充
  - `void *mask_addr`: 设备端掩码矩阵内存地址
  - `void *value_addr`: 设备端负无穷值内存地址
  - `uint64_t workspacesize`: 工作空间大小(取softmax和masked_fill的最大值)
  - `aclOpExecutor *executor`: 可复用的softmax算子执行器
- **生命周期**:
  - 在`Descriptor::create`中分配资源
  - 析构函数释放ACL描述符、设备内存和执行器
  - 使用RAII模式管理ACL资源

### `Descriptor`
- **位置**: 通过`causal_softmax.h`中的`DESCRIPTOR(ascend)`宏定义
- **主要功能**: 公共接口类，继承自`InfiniopDescriptor`，提供因果softmax算子创建和执行接口
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 构建算子描述符
    - 验证输入输出张量形状和类型一致性
    - 创建ACL张量描述符，处理batch维度和stride信息
    - 在CPU端构建因果掩码矩阵(上三角区域为1)
    - 将掩码矩阵和负无穷值拷贝到设备内存
    - 查询softmax和masked_fill算子所需工作空间大小
    - 设置softmax执行器为可复用模式(`aclSetAclOpExecutorRepeatable`)
    - 时间复杂度: O(seq_len × total_seq_len) 用于掩码构建，O(1) 算子查询
  - `calculate(workspace, workspace_size, y, x, stream)`: 执行因果softmax计算
    - 验证工作空间大小
    - **第一阶段**: 执行原地masked_fill操作，将上三角区域(未来位置)填充为负无穷
      - 使用`AclSetTensorAddr`动态绑定输入张量地址
      - 调用`aclnnInplaceMaskedFillTensor`完成掩码
    - **第二阶段**: 在最后一个维度(dim=2)上执行softmax归一化
      - 调用`aclnnSoftmax`对masked后的张量进行指数归一化
    - 时间复杂度: O(batch_size × seq_len × total_seq_len)
- **继承结构**:
  - 继承`InfiniopDescriptor`基类
  - 包含`Opaque`不透明指针实现PIMPL模式
  - 持有`CausalSoftmaxInfo`存储张量元信息

## 3. API接口

```cpp
// 创建因果softmax算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                // Ascend设备句柄
    Descriptor **desc_ptr,                  // 输出: 算子描述符指针
    infiniopTensorDescriptor_t y_desc,      // 输出张量描述符 (batch, seq_len, total_seq_len)
    infiniopTensorDescriptor_t x_desc       // 输入张量描述符 (batch, seq_len, total_seq_len)
);
// 返回: INFINI_STATUS_SUCCESS / 错误码(类型不匹配/形状非法)

// 执行因果softmax计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                        // 设备工作空间指针
    size_t workspace_size,                  // 工作空间大小(字节)
    void *y,                                // 输出张量设备地址
    const void *x,                          // 输入张量设备地址
    void *stream                            // Ascend ACL stream句柄
) const;
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE
```

## 4. 使用示例

```cpp
// 初始化Ascend设备和句柄
device::ascend::Handle *handle;
infiniopHandle_t handle_t = reinterpret_cast<infiniopHandle_t>(handle);

// 准备张量描述符 (batch_size=2, seq_len=128, total_seq_len=256)
std::vector<int64_t> shape = {2, 128, 256};
infiniopTensorDescriptor_t x_desc = new TensorDescriptor(ACL_FLOAT16, shape);
infiniopTensorDescriptor_t y_desc = new TensorDescriptor(ACL_FLOAT16, shape);

// 创建算子
op::causal_softmax::ascend::Descriptor *desc;
infiniStatus_t status = op::causal_softmax::ascend::Descriptor::create(
    handle_t, &desc, y_desc, x_desc);

// 分配工作空间和设备内存
void *workspace = nullptr;
aclrtMalloc(&workspace, desc->workspaceSize(), ACL_MEM_MALLOC_HUGE_FIRST);

void *d_x, *d_y;
aclrtMalloc(&d_x, x_desc->nbytes(), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_y, y_desc->nbytes(), ACL_MEM_MALLOC_HUGE_FIRST);

// 拷贝输入数据到设备
aclrtMemcpy(d_x, x_desc->nbytes(), host_x, x_desc->nbytes(), ACL_MEMCPY_HOST_TO_DEVICE);

// 执行因果softmax (两阶段: 掩码填充 -> softmax归一化)
aclrtStream stream;
aclrtCreateStream(&stream);
desc->calculate(workspace, desc->workspaceSize(), d_y, d_x, stream);
aclrtSynchronizeStream(stream);

// 拷贝结果回主机
aclrtMemcpy(host_y, y_desc->nbytes(), d_y, y_desc->nbytes(), ACL_MEMCPY_DEVICE_TO_HOST);

// 清理资源
delete desc;
aclrtFree(d_x);
aclrtFree(d_y);
aclrtFree(workspace);
```

## 5. 实现细节

### 内存管理
- **设备内存分配**: 使用`aclrtMalloc`配合`ACL_MEM_MALLOC_HUGE_FIRST`策略分配大页内存
- **掩码矩阵**: 在CPU端构建bool型矩阵(seq_len × total_seq_len)，通过`aclrtMemcpy`拷贝到设备
  - 上三角区域(total_seq_len - seq_len + i + 1 至 total_seq_len)设置为1(真值)
  - 下三角和主对角线区域保持0(假值)
- **负无穷值**: 根据数据类型在设备端分配4字节标量
  - FP16: `0xfc00` (IEEE 754半精度负无穷)
  - FP32: `0xff800000` (IEEE 754单精度负无穷)
- **RAII释放**: `Opaque`析构函数自动调用`aclrtFree`释放设备内存

### 并发与执行
- **Stream并发**: calculate方法接受外部传入的ACL stream，支持多stream并行执行
- **执行器复用**: softmax算子执行器通过`aclSetAclOpExecutorRepeatable`设置为可复用，避免每次调用重新创建
- **原地操作**: masked_fill使用原地(inplace)修改输入张量，减少内存拷贝开销

### 性能优化
- **工作空间复用**: workspace_size取softmax和masked_fill的最大值，单个buffer复用
- **算子融合**: 避免手动kernel编写，直接使用CANN库函数优化实现
- **内存布局**: 支持非连续张量(通过stride参数处理广播和转置场景)
- **延迟绑定**: calculate阶段通过`AclSetTensorAddr`动态绑定实际内存地址

### 错误处理
- **宏定义检查**: 使用`CHECK_ACL`宏包装ACL API调用，自动将错误码转换为`infiniStatus_t`
- **类型验证**: create阶段检查输入输出dtype一致性(FP16/BF16/FP32)
- **形状约束**:
  - 张量维度必须为2D或3D
  - 要求 total_seq_len >= seq_len (否则返回`INFINI_STATUS_BAD_TENSOR_SHAPE`)
- **工作空间校验**: calculate阶段验证workspace_size是否足够

### 依赖关系
- **CANN算子库**:
  - `aclnn_masked_fill_tensor.h`: 提供因果掩码填充功能
  - `aclnn_softmax.h`: 提供softmax归一化功能
- **公共基础设施**:
  - `../causal_softmax.h`: 定义Descriptor类宏模板
  - `info.h`: 提供`CausalSoftmaxInfo`元数据验证
  - `../../../devices/ascend/common_ascend.h`: Ascend设备通用工具函数(如`toAclDataType`, `AclSetTensorAddr`)

### 设计模式
- **PIMPL (Pointer to Implementation)**: 通过`Opaque`结构体隐藏ACL实现细节
- **宏驱动接口**: `DESCRIPTOR(NAMESPACE)`宏为不同后端生成统一接口类
- **RAII资源管理**: 析构函数自动释放ACL描述符和设备内存
- **策略模式**: 同一causal_softmax算子支持多种硬件后端(cuda/cpu/ascend等)
