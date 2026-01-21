# RMSNorm Bang实现核心文档 (RMS Norm Bang Implementation Core Documentation)

本模块实现了针对寒武纪(Cambricon) MLU硬件的Root Mean Square Normalization (RMS归一化)算子。该实现针对Bang语言进行了深度优化,支持混合精度计算(FP16/FP32/BF16),并采用双阶段算法(先计算平方和再归一化)以最大化硬件利用率和计算效率。

## 1. 模块结构 (Module Structure)

- **`rms_norm_bang.h`**: 头文件,包含DESCRIPTOR宏定义,声明bang命名空间下的Descriptor类
- **`rms_norm_bang.mlu`**: 核心实现文件,包含设备端核函数和主机端调度逻辑

## 2. 核心类与函数 (Core Classes and Functions)

### `rmsnorm` 核函数 (Kernel Function)
- **位置**: `rms_norm_bang.mlu:8-215`
- **函数签名**:
```cpp
template <typename T, typename Tw>
__mlu_global__ void rmsnorm(T *output, const T *input, const Tw *weight,
                            size_t *shape, ptrdiff_t *output_strides, ptrdiff_t *input_strides,
                            float epsilon, int num_dims, int norm_dim_size)
```
- **主要功能**: 在MLU设备上执行RMS归一化操作,针对每个独立向量计算归一化因子并应用权重缩放
- **模板参数**:
  - `T`: 输入/输出数据类型 (支持 `half`, `float`, `bfloat16_t`)
  - `Tw`: 权重数据类型 (支持 `half`, `float`, `bfloat16_t`)
- **核心算法**:
  1. **任务分配策略** (第18-22行): 使用负载均衡算法将批次任务分配给各计算核心
    - 计算 `remaining_tasks = batch_volume % taskDim`
    - 计算每个核心的基础任务数 `base_tasks_per_core = batch_volume / taskDim`
    - 前 `remaining_tasks` 个核心多分配1个任务,其余核心分配基础任务数
    - 时间复杂度: O(1) 任务分配

  2. **批处理大小优化** (第24-34行): 动态计算最优批处理大小以最大化NRAM利用率
    - 小向量 (≤64元素): 一次处理整个向量
    - 大向量 (>64元素): 使用公式 `max_batch_size = (NRAM_MAX_SIZE - 256) / (2*sizeof(T) + sizeof(Tw) + sizeof(float))`
    - 强制64元素对齐以利用SIMD指令: `max_batch_size = (max_batch_size / 64) * 64`

  3. **NRAM内存布局** (第38-43行): 精心设计的内存分区以减少bank冲突
    ```
    nram_buffer布局:
    [reduction_buffer: 128字节] [input_cache: max_batch_size*sizeof(T)]
    [weight_cache: max_batch_size*sizeof(Tw)] [float_buffer: max_batch_size*sizeof(float)]
    [weight_float_buffer: max_batch_size*sizeof(float)]
    ```

  4. **平方和计算阶段** (第61-120行): 采用两种优化路径
    - **小向量路径** (≤128元素): 一次性加载全向量,直接转换为float并平方,使用标量累加
    - **大向量路径** (>128元素): 分块处理,使用 `op::common_bang::reduce_op::sumInternal` 进行向量化归约
      - 批次大小≥128时使用硬件归约指令
      - 批次大小<128时回退到标量累加

  5. **归一化因子计算** (第122-124行):
    ```cpp
    float rms_value = sqrtf(sum_squared / vector_size + epsilon);
    float inv_rms = 1.0f / rms_value;  // 预计算倒数,将除法转换为乘法
    ```

  6. **应用权重与归一化** (第126-213行): 双阶段处理
    - **小向量路径** (第127-164行): 一次性加载全部数据,执行向量操作
    - **大向量路径** (第166-212行): 分块加载,逐块处理
    - 操作序列: `input → float → mul(weight) → mul_scalar(inv_rms) → output_type`

### `rmsnormUnion` 主机端调度函数
- **位置**: `rms_norm_bang.mlu:217-259`
- **函数签名**:
```cpp
template <typename T, typename Tw>
void rmsnormUnion(void *workspace, int core_per_cluster, int cluster_count, cnrtQueue_t queue,
                  void *y, const void *x, const void *w, const size_t *shape,
                  const ptrdiff_t *y_strides, const ptrdiff_t *x_strides,
                  float eps, int ndim)
```
- **主要功能**: 配置核函数启动参数并异步拷贝元数据到设备
- **关键逻辑**:
  1. **核函数配置** (第219-226行):
     ```cpp
     kernel_dim.x = core_per_cluster;      // 每个集群的核心数
     kernel_dim.y = cluster_count;         // 集群数量
     kernel_dim.z = 1;
     kernel_type = CNRT_FUNC_TYPE_UNION1;  // Union1类型kernel
     ```
  2. **对齐尺寸计算** (第228-238行): 计算下一轮2的幂次,确保满足 `dim_s >= reduce_num` (128字节对齐要求)
  3. **异步内存拷贝** (第251-253行): 使用 `cnrtMemcpyAsync` 将shape和stride信息传输到设备
  4. **核函数启动** (第256行): `<<<kernel_dim, kernel_type, queue>>>` 三重启动语法
  5. **同步** (第258行): `cnrtQueueSync(queue)` 确保计算完成

### `Descriptor::create` 算子描述符创建函数
- **位置**: `rms_norm_bang.mlu:271-293`
- **功能**: 验证输入参数并创建RMSNorm算子描述符
- **验证逻辑**:
  1. 调用 `RMSNormInfo::create` 进行类型和形状验证
  2. 计算workspace大小: `workspace_size = ndim * (sizeof(size_t) + 2 * sizeof(ptrdiff_t))`
  3. 存储设备类型和设备ID
- **错误处理**: 使用 `CHECK_RESULT(result)` 宏进行错误传播

### `Descriptor::calculate` 算子执行函数
- **位置**: `rms_norm_bang.mlu:295-338`
- **功能**: 根据数据类型分发到对应的模板实例化
- **类型分发矩阵** (第302-335行): 支持9种类型组合
  - FP16输入 + FP16/FP32/BF16权重
  - FP32输入 + FP32/FP16/BF16权重
  - BF16输入 + BF16/FP32/FP16权重
- **执行流程**:
  1. 从opaque中获取核心数和集群数
  2. 根据 `_info.atype` 和 `_info.wtype` 进行三重if-else分发
  3. 调用对应的 `rmsnormUnion<T, Tw>` 实例

### `Descriptor` 类
- **位置**: 通过 `DESCRIPTOR(bang)` 宏在 `rms_norm.h` 中定义
- **继承关系**: `public InfiniopDescriptor`
- **成员变量**:
  - `Opaque *_opaque`: 设备句柄内部状态 (智能指针管理)
  - `RMSNormInfo _info`: 张量形状和类型信息
  - `size_t _workspace_size`: 设备端临时存储大小
- **生命周期**: RAII管理,析构函数自动删除 `_opaque`
- **关键方法**:
  - `workspaceSize()`: 返回所需workspace大小
  - `create(...)`: 工厂方法创建描述符实例
  - `calculate(...)`: 执行RMS归一化计算

## 3. API接口 (API Interface)

### 公共API
```cpp
namespace op::rms_norm::bang {
    class Descriptor final : public InfiniopDescriptor {
    public:
        // 获取所需workspace大小
        size_t workspaceSize() const;

        // 创建算子描述符
        static infiniStatus_t create(
            infiniopHandle_t handle,                  // 设备句柄
            Descriptor **desc_ptr,                    // 输出描述符指针
            infiniopTensorDescriptor_t y_desc,        // 输出张量描述符
            infiniopTensorDescriptor_t x_desc,        // 输入张量描述符
            infiniopTensorDescriptor_t w_desc,        // 权重张量描述符
            float epsilon);                           // 数值稳定性常数

        // 执行RMS归一化
        infiniStatus_t calculate(
            void *workspace,                          // 设备端临时内存
            size_t workspace_size,                    // workspace大小
            void *y,                                  // 输出数据指针
            const void *x,                            // 输入数据指针
            const void *w,                            // 权重数据指针
            void *stream) const;                      // CUDA/Bang流
    };
}
```

### 设备端内核接口
```cpp
// 核心设备端函数,由rmsnormUnion间接调用
template <typename T, typename Tw>
__mlu_global__ void rmsnorm(
    T *output,                      // 输出缓冲区 [shape[0]*...*shape[ndim-2], shape[ndim-1]]
    const T *input,                 // 输入缓冲区,形状同输出
    const Tw *weight,               // 权重向量 [shape[ndim-1]]
    size_t *shape,                  // 张量形状数组 (设备端)
    ptrdiff_t *output_strides,      // 输出步长数组 (设备端)
    ptrdiff_t *input_strides,       // 输入步长数组 (设备端)
    float epsilon,                  // 防止除零的小常数
    int num_dims,                   // 张量维度数 (2或3)
    int norm_dim_size);             // 归一化维度的对齐大小
```

## 4. 使用示例 (Usage Example)

```cpp
#include "infiniop.h"
#include "infiniop_ops/rms_norm/bang/rms_norm_bang.h"

// 初始化
infiniopHandle_t handle;
infiniopCreateHandle(&handle, 0);  // 设备ID=0

// 配置张量描述符 (2D张量: [batch_size, hidden_dim])
constexpr size_t batch_size = 128;
constexpr size_t hidden_dim = 4096;

infiniopTensorDescriptor_t x_desc, y_desc, w_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 2, {batch_size, hidden_dim});
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 2, {batch_size, hidden_dim});
infiniopCreateTensorDescriptor(&w_desc, INFINI_DTYPE_F32, 1, {hidden_dim});

// 创建RMSNorm算子描述符
op::rms_norm::bang::Descriptor *rmsnorm_desc;
float epsilon = 1e-6f;
auto status = op::rms_norm::bang::Descriptor::create(
    handle, &rmsnorm_desc, y_desc, x_desc, w_desc, epsilon);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
    return;
}

// 分配设备内存
half *d_x, *d_y;
float *d_w;
cnrtMalloc(&d_x, batch_size * hidden_dim * sizeof(half));
cnrtMalloc(&d_y, batch_size * hidden_dim * sizeof(half));
cnrtMalloc(&d_w, hidden_dim * sizeof(float));

// 拷贝输入数据
cnrtMemcpy(d_x, h_x, batch_size * hidden_dim * sizeof(half), cnrtMemcpyHostToDev);
cnrtMemcpy(d_w, h_w, hidden_dim * sizeof(float), cnrtMemcpyHostToDev);

// 分配workspace
size_t workspace_size = rmsnorm_desc->workspaceSize();
void *d_workspace;
cnrtMalloc(&d_workspace, workspace_size);

// 创建Bang流
cnrtQueue_t stream;
cnrtQueueCreate(&stream);

// 执行RMS归一化
status = rmsnorm_desc->calculate(d_workspace, workspace_size, d_y, d_x, d_w, stream);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理执行错误
}

// 同步并取回结果
cnrtQueueSync(stream);
cnrtMemcpy(h_y, d_y, batch_size * hidden_dim * sizeof(half), cnrtMemcpyDevToHost);

// 清理资源
delete rmsnorm_desc;
cnrtFree(d_x); cnrtFree(d_y); cnrtFree(d_w); cnrtFree(d_workspace);
cnrtQueueDestroy(stream);
infiniopDestroyHandle(handle);
```

## 5. 实现细节 (Implementation Details)

### 内存管理策略
- **NRAM池化**: 使用全局静态NRAM缓冲区 `__nram__ char nram_buffer[NRAM_MAX_SIZE]`,大小为240KB (1024*240字节)
- **动态分区**: 根据向量大小动态调整 `input_cache`, `weight_cache`, `float_buffer` 的大小
- **对齐策略**:
  - 批处理大小强制64元素对齐以利用SIMD向量指令
  - 减少缓冲区128字节对齐 (128/sizeof(float) = 32个float)
  - Shape/Stride信息在设备端需要4字节对齐

### 并发与线程安全
- **并行策略**: 使用Union1 Kernel类型,在多个计算核心间并行处理批次维度
- **负载均衡**: 采用不均匀任务分配算法,前 `batch_volume % taskDim` 个核心额外处理1个任务,最大化核心利用率
- **同步机制**:
  - 主机端使用 `cnrtQueueSync` 确保kernel完成
  - 使用异步拷贝 (`cnrtMemcpyAsync`) 隐藏数据传输延迟
- **线程安全性**: Descriptor对象本身不是线程安全的,但每个stream可以独立执行不同的calculate调用

### 性能优化技术
1. **类型特化优化**: 使用 `if constexpr` 编译期分支消除,避免运行时类型判断开销
   ```cpp
   if constexpr (std::is_same<T, half>::value) {
       __bang_half2float(float_buffer, input_cache, vector_size);
   } else if constexpr (std::is_same<T, bfloat16_t>::value) {
       __bang_bfloat162float(float_buffer, input_cache, vector_size);
   }
   ```

2. **向量化归约**: 使用 `__bang_sumpool` + `__bang_reduce_sum` 组合进行高效求和
   - `__bang_sumpool`: 在宽度维度上进行向量求和
   - `__bang_reduce_sum`: 对128字节数据进行标量归约
   - 仅在批次大小≥4 * batch_size时启用,否则回退到标量循环

3. **除法转乘法**: 预计算 `inv_rms = 1.0f / rms_value`,在后续归一化中使用乘法替代除法
   ```cpp
   __bang_mul_scalar(float_buffer, float_buffer, inv_rms, vector_size);
   ```

4. **批处理分块**: 对大向量 (>128元素) 采用分块策略,每块大小自适应NRAM容量

5. **NRAM利用率优化**: 通过 `(NRAM_MAX_SIZE - 256) / (2*sizeof(T) + sizeof(Tw) + sizeof(float))` 公式精确计算最大批处理大小

### 错误处理
- **类型验证**: `RMSNormInfo::create` 验证输入/输出/权重类型兼容性
  - FP16/BF16激活可以搭配FP16/FP32/BF16权重
  - FP32激活必须搭配FP32权重
- **形状验证**: 支持2D [batch, dim] 和3D [batch, nhead, dim] 张量
- **步长验证**: 确保最后一维连续 (stride=1)
- **返回值传播**: 使用 `CHECK_RESULT` 宏和 `INFINI_STATUS_*` 错误码

### 依赖关系
- **硬件依赖**:
  - Cambricon MLU架构 (寒武纪MLU芯片)
  - BANG语言特性 (NRAM, GDRAM, 核函数启动语法)
- **算子依赖**:
  - `op::common_bang::reduce_op::sumInternal`: 向量化求和归约
  - `device::bang::Handle::Internal`: 设备句柄管理 (获取核心数/集群数)
- **外部库依赖**:
  - `cnrt.h`: Cambricon运行时API
  - `cnnl.h`: Cambricon神经网络计算库
  - `bang.h`: BANG内建函数 (__bang_*, __memcpy, __mlu_global__, etc.)

### 设计模式
- **策略模式**: 通过模板参数 `<T, Tw>` 支持不同的数据类型组合策略
- **工厂模式**: `Descriptor::create` 作为工厂方法构造描述符
- **RAII**: Descriptor析构函数自动释放Opaque资源
- **Pimpl模式**: 使用 `struct Opaque` 封装设备相关实现细节
- **宏生成**: `DESCRIPTOR(bang)` 宏自动生成Descriptor类定义,减少样板代码

### 算法复杂度
- **时间复杂度**: O(N),其中N为输入张量元素总数
  - 平方和计算阶段: O(N),每个元素访问一次
  - 归一化应用阶段: O(N),每个元素访问一次
  - 总计: O(2N) = O(N)
- **空间复杂度**: O(D + S),其中D为归一化维度大小,S为shape/stride存储开销
  - NRAM缓冲区: O(max_batch_size),通常为64-256KB
  - Workspace: O(ndim * (sizeof(size_t) + 2*sizeof(ptrdiff_t))) ≈ O(24*ndim)字节
- **批处理效率**: 对大向量,批处理效率 = max_batch_size / vector_size,接近100%利用率
