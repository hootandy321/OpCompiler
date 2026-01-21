# GEMM Ascend 算子核心实现文档

本模块实现了华为昇腾(Ascend)AI处理器上的通用矩阵乘法(GEMM)算子，基于华为CANN(Compute Architecture for Neural Networks)计算框架的aclnn API，提供高性能的FP16和FP32矩阵乘法运算，支持批量矩阵乘法和alpha/beta缩放参数。

## 1. 模块结构

- **`gemm_ascend.h`**: 算子描述符的头文件声明，使用宏定义生成统一的Descriptor接口
- **`gemm_ascend.cc`**: 核心实现文件，包含算子创建、执行器和缓存机制

## 2. 核心类

### `Descriptor`
- **位置**: `gemm_ascend.h` (宏展开定义), `gemm_ascend.cc` (具体实现)
- **主要功能**: 封装昇腾GEMM算子的所有元数据和执行状态，提供统一的硬件无关接口
- **继承关系**: 继承自 `InfiniopDescriptor` (位于 `../../operator.h`)
- **生命周期**: 由 `Descriptor::create()` 工厂方法创建，析构时自动清理ACL资源

#### 关键成员变量
- **`_opaque`**: `Opaque*` 类型，PImpl模式的不透明指针，隐藏硬件相关实现细节
- **`_dtype`**: `infiniDtype_t`，存储矩阵元素的数据类型(F16或F32)
- **`_info`**: `MatmulInfo`，存储矩阵维度信息(BMNK)、步长(stride)、批量大小等
- **`_workspace_size`**: `size_t`，ACL算子执行所需的最大工作空间大小

### `Descriptor::Opaque`
- **位置**: `gemm_ascend.cc` 第28-46行
- **主要功能**: 封装所有华为ACL相关的硬件特定类型和状态
- **设计模式**: PImpl (Pointer to Implementation) 模式，隔离硬件相关类型

#### 核心成员
- **`c, a, b`**: `aclnnTensorDescriptor_t` 类型，分别对应输出矩阵C、输入矩阵A和矩阵B的ACL张量描述符
- **`mt`**: `int8_t` 类型，cubeMathType参数，控制昇腾AI Core的立方计算单元精度模式
  - 值为1表示默认模式(参见[华为官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md))
- **`lookup`**: `std::unordered_map<std::pair<float, float>, aclOpExecutor *, FloatPairHash, FloatPairEqual>`
  - 键: `(alpha, beta)` 参数对
  - 值: 对应的ACL操作执行器(aclOpExecutor*)
  - **作用**: 缓存不同alpha/beta组合的预编译执行器，避免运行时重复编译开销

#### 析构逻辑
```cpp
~Opaque() {
    delete c;  // 释放ACL张量描述符
    delete a;
    delete b;
    for (auto &item : lookup) {
        aclDestroyAclOpExecutor(item.second);  // 销毁所有缓存的执行器
    }
    lookup.clear();
}
```

### `FloatPairHash` 和 `FloatPairEqual`
- **位置**: `gemm_ascend.cc` 第10-24行
- **主要功能**: 为 `std::pair<float, float>` 提供哈希和相等比较，用作unordered_map的键类型

#### FloatPairHash 哈希算法
```cpp
struct FloatPairHash {
    size_t operator()(const std::pair<float, float> &p) const {
        uint64_t combined;
        // 将两个float的位模式拼接成一个64位整数
        std::memcpy(&combined, &p.first, sizeof(float));    // 前32位存alpha
        std::memcpy(reinterpret_cast<char*>(&combined) + sizeof(float), &p.second, sizeof(float));  // 后32位存beta
        return std::hash<uint64_t>()(combined);  // 使用标准库哈希函数
    }
};
```
- **技术细节**:
  - 使用 `memcpy` 而非强制类型转换，避免严格别名(strict aliasing)规则的未定义行为
  - 直接使用float的位模式进行哈希，对于相同的浮点值(包括NaN的位模式)保证哈希一致
  - 空间复杂度: O(1)
  - 时间复杂度: O(1)

## 3. API接口

### `Descriptor::create()`
```cpp
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,           // 昇腾设备句柄
    Descriptor **desc_ptr,              // 输出参数：创建的描述符指针
    infiniopTensorDescriptor_t c_desc,  // 输出矩阵C的描述符
    infiniopTensorDescriptor_t a_desc,  // 输入矩阵A的描述符
    infiniopTensorDescriptor_t b_desc   // 输入矩阵B的描述符
);
```
- **功能**: 创建GEMM描述符，初始化ACL张量描述符和预编译常用执行器
- **返回值**:
  - `INFINI_STATUS_SUCCESS`: 成功
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型(仅支持F16/F32)
  - 其他错误: 从MatmulInfo::create传播
- **执行流程**:
  1. 验证数据类型必须是FP16或FP32
  2. 调用 `MatmulInfo::create()` 解析张量形状，验证维度兼容性，计算矩阵步长
  3. 创建三个ACL张量描述符，传入形状、步长和数据类型
  4. **预编译优化**: 预先为两种常用alpha/beta组合编译执行器并缓存:
     - `(1.0, 0.0)`: 标准GEMM (C = alpha * A @ B)
     - `(1.0, 1.0)`: 累加GEMM (C = alpha * A @ B + beta * C)
  5. 设置执行器为可重复模式(`aclSetAclOpExecutorRepeatable`)
  6. 计算工作空间大小(取两种配置的最大值)

### `Descriptor::calculate()`
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,              // 设备内存工作空间指针
    size_t workspaceSize_,        // 工作空间大小
    void *c,                      // 输出矩阵C的设备指针
    float beta,                   // beta缩放参数
    const void *a,                // 输入矩阵A的设备指针
    const void *b,                // 输入矩阵B的设备指针
    float alpha,                  // alpha缩放参数
    void *stream                  // 昇腾Stream句柄
) const;
```
- **功能**: 执行GEMM计算 C = alpha * A @ B + beta * C
- **返回值**:
  - `INFINI_STATUS_SUCCESS`: 计算成功
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间不足
  - 其他ACL错误: 通过CHECK_ACL宏传播
- **执行流程**:
  1. **执行器查找**: 在 `_opaque->lookup` 中查找当前(alpha, beta)组合
  2. **懒编译**: 如果未找到，调用 `aclnnGemmGetWorkspaceSize()` 实时编译并缓存
  3. **工作空间验证**: 检查用户提供的工作空间是否满足需求
  4. **批量计算循环**:
     - 对每个batch样本，更新执行器中的张量地址指针
     - 调用 `aclnnGemm()` 在AI Core上执行矩阵乘法
  5. 返回成功状态

## 4. 使用示例

```cpp
// 示例：在昇腾NPU上执行批量GEMM (C = 1.0 * A @ B + 0.5 * C)

#include "infiniop/ops/gemm/gemm_ascend.h"

// 1. 准备张量描述符 (假设已初始化)
infiniopTensorDescriptor_t c_desc, a_desc, b_desc;
// ... 设置形状为 [batch, M, N], [batch, M, K], [batch, K, N]

// 2. 创建GEMM描述符
op::gemm::ascend::Descriptor* gemm_desc;
infiniStatus_t status = op::gemm::ascend::Descriptor::create(
    ascend_handle,    // 昇腾设备句柄
    &gemm_desc,       // 输出描述符指针
    c_desc,           // 输出矩阵 [batch, M, N]
    a_desc,           // 输入矩阵A [batch, M, K]
    b_desc            // 输入矩阵B [batch, K, N]
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 分配工作空间
size_t workspace_size = gemm_desc->workspaceSize();
void* workspace;
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);

// 4. 准备设备内存
void* d_a, *d_b, *d_c;
aclrtMalloc(&d_a, a_size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_b, b_size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_c, c_size, ACL_MEM_MALLOC_HUGE_FIRST);
// ... 将数据从主机拷贝到设备

// 5. 执行GEMM计算 (alpha=1.0, beta=0.5)
status = gemm_desc->calculate(
    workspace, workspace_size,  // 工作空间
    d_c, 0.5f,                  // 输出矩阵和beta
    d_a, d_b, 1.0f,             // 输入矩阵和alpha
    ascend_stream               // 昇腾stream
);

// 6. 同步并清理资源
aclrtSynchronizeStream(ascend_stream);
// ... 将结果从设备拷贝回主机
aclrtFree(d_a);
aclrtFree(d_b);
aclrtFree(d_c);
aclrtFree(workspace);
delete gemm_desc;
```

**批量矩阵乘法示例** (batch=32, M=128, N=256, K=512):
```cpp
// 输入: A[32, 128, 512], B[32, 512, 256]
// 输出: C[32, 128, 256]
// 内部循环会对32个样本依次调用ACL GEMM kernel
// 每个样本独立计算，充分利用昇腾AI Core的并行能力
```

## 5. 实现细节

### 内存管理
- **张量描述符生命周期**: 由 `Opaque` 析构函数统一释放，使用 `delete` 销毁 `aclnnTensorDescriptor` 对象
- **执行器缓存管理**: 使用 `std::unordered_map` 存储，析构时遍历调用 `aclDestroyAclOpExecutor`
- **设备内存**: 由调用方管理，算子仅接收设备指针并设置到执行器中
- **RAII原则**: 所有ACL资源都在描述符析构时自动释放，防止资源泄漏

### 并发与线程安全
- **无状态设计**: `Descriptor` 对象本身不可变(const成员函数)，所有状态存储在 `_opaque` 中
- **执行器重入性**: 通过 `aclSetAclOpExecutorRepeatable(executor)` 设置执行器为可重复模式
  - 允许多个stream同时使用同一个执行器
  - 同一stream上多次调用安全
- **哈希表并发**: `lookup` 哈希表在create()时初始化，calculate()中可能新增条目
  - **警告**: 如果多个线程同时调用同一个Descriptor对象的calculate()，会有竞争条件
  - **使用建议**: 每个线程使用独立的Descriptor实例，或外部加锁保护

### 性能优化
- **执行器缓存**: 预编译常用(alpha, beta)组合，避免每次调用时的编译开销
  - 缓存命中: O(1) 哈希查找
  - 缓存未命中: 首次调用时会触发ACL JIT编译，后续调用直接使用缓存
- **批量处理**: 支持batch维度的循环展开，每个batch独立调用ACL kernel
- **工作空间复用**: 计算所有alpha/beta组合的最大工作空间需求，一次性分配
- **零拷贝设置**: 使用 `AclSetTensorAddr` 直接修改执行器中的张量地址，避免重新创建执行器

### 错误处理
- **CHECK_ACL宏**: 定义在 `common_ascend.h` 中，检查ACL API返回值
  - 成功时返回 `ACL_SUCCESS`
  - 失败时调用 `GetRecentErrMsg()` 打印华为错误消息，然后返回错误码
- **类型安全**: 严格检查数据类型，仅支持FP16和FP32
- **维度验证**: 通过 `MatmulInfo::create()` 验证矩阵形状兼容性
  - 检查 M, N, K 维度匹配
  - 检查batch维度一致
  - 检查stride合法性(至少有一个维度的stride为1)

### 依赖关系
- **华为CANN框架**:
  - `aclnnop/aclnn_matmul.h`: 矩阵乘法高层API
  - `aclnnop/level2/aclnn_gemm.h`: GEMM算子API
  - `acl/acl.h`: 核心ACL运行时
  - `aclnn/acl_meta.h`: 张量元数据描述
- **内部模块**:
  - `../gemm.h`: DESCRIPTOR宏定义，生成统一的Descriptor类
  - `../info.h`: MatmulInfo和BlasMatrix结构，处理张量形状解析
  - `../../../devices/ascend/common_ascend.h`: ACL通用工具和类型定义
  - `../../operator.h`: InfiniopDescriptor基类
  - `../../../utils.h`: 通用工具类和错误处理宏

### 设计模式
- **PImpl (Pointer to Implementation)**: `Opaque` 结构体隐藏ACL相关类型
  - 优点: 头文件 `gemm_ascend.h` 不包含ACL特定的头文件，保持接口硬件无关
  - 实现细节: 仅在 `.cc` 文件中定义 `struct Descriptor::Opaque`
- **工厂模式**: `Descriptor::create()` 静态方法封装复杂的创建逻辑
- **策略模式**: 通过 `DESCRIPTOR(ascend)` 宏生成不同硬件的Descriptor类，但共享相同接口
- **缓存模式**: `lookup` 哈希表实现执行器缓存，避免重复编译

### 算法复杂度
- **创建阶段**:
  - 张量描述符创建: O(1)
  - 预编译执行器: O(ACL编译时间)，通常为毫秒级
  - 哈希表插入: O(1) 平均
- **计算阶段**:
  - 执行器查找: O(1) 平均哈希查找
  - 执行器创建(缓存未命中): O(ACL编译时间)，一次性开销
  - GEMM计算: O(batch * M * N * K) 浮点运算
    - 由昇腾AI Core硬件并行加速，实际时间取决于NPU数量和频率
- **空间复杂度**:
  - 描述符内存: O(1) 固定大小
  - 执行器缓存: O(n) 其中n为不同(alpha, beta)组合数量
  - 工作空间: 由ACL内部决定，通常为O(M*K + K*N)量级

### 硬件特定行为
- **昇腾AI Core**: 使用华为自研的Da Vinci架构立方计算单元
- **cubeMathType(mt)**: 控制计算精度模式
  - 默认值为1，启用标准精度计算
  - 可能影响FP16累积精度和性能
- **流水线并行**: 批量GEMM通过循环依次提交，NPU内部可能自动流水线化
- **内存访问**: 建议使用 `ACL_MEM_MALLOC_HUGE_FIRST` 获取大页内存，提高带宽利用率
