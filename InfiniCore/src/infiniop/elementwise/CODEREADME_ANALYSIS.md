# 📂 目录: elementwise 逐元素运算架构全景

## 1. 子系统职责

`elementwise` 目录是 InfiniOp 算子库的核心子系统,负责实现所有逐元素(Element-wise)运算操作。该子系统遵循统一的接口抽象,支持多硬件后端加速,包括 CPU、GPU(NVIDIA/CUDA)、国产加速卡(昆仑、摩尔线程、天数智芯、寒武纪)等。

**核心设计理念**:
- **接口统一**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏实现跨硬件的统一算子描述符
- **元数据封装**: `ElementwiseInfo` 结构体集中管理张量形状、步长、布局、广播等元信息
- **硬件抽象**: 每个硬件后端独立实现 DeviceImpl,实现计算内核与上层接口解耦
- **广播支持**: 自动处理张量广播机制,支持不同形状张量的逐元素运算

## 2. 模块导航

* **📂 bang**:
    * *功能*: 寒武纪(Cambricon) MLU 硬件后端实现
    * *职责*: 基于 BangC 语言实现逐元素运算的 BANG 版本内核,提供 `elementwise_bang_api.h`、`elementwise_bang.h`、`elementwise_bang_kernel.h` 三层接口封装

* **📂 cpu**:
    * *功能*: CPU 通用后端实现
    * *职责*: 基于 C++ 实现跨平台的 CPU 逐元素运算,提供 `elementwise_cpu.h` 单文件实现,作为基准实现和 fallback 方案

* **📂 kunlun**:
    * *功能*: 昆仑(Kunlun) 硬件后端实现
    * *职责*: 基于昆仑 XPU 加速卡实现逐元素运算,提供 `elementwise_kunlun_api.h` 和 `elementwise_kunlun.h` 两层封装

* **📂 metax**:
    * *功能*: 天数智芯(Metax) 硬件后端实现
    * *职责*: 基于 Metax GPU 实现逐元素运算,提供 `elementwise_metax_api.h` 和 `elementwise_metax.h` 两层封装

* **📂 moore**:
    * *功能*: 摩尔线程(Moore Threads) 硬件后端实现
    * *职责*: 基于 MUSA 架构实现逐元素运算,提供 `elementwise_moore_api.h` 和 `elementwise_moore.h` 两层封装

* **📂 nvidia**:
    * *功能*: NVIDIA CUDA 硬件后端实现
    * *职责*: 基于 CUDA 实现高性能逐元素运算内核,提供 `elementwise_nvidia_api.cuh` 和 `elementwise_nvidia.cuh` 两层封装,支持 GPU 并行加速

## 3. 架构逻辑图解

### 3.1 元数据流

```
Tensor Descriptors (输入/输出)
    ↓
ElementwiseInfo::create()
    ├─ 提取 shape、strides、ndim
    ├─ 检测连续性(isContiguous)
    ├─ 识别广播维度(hasBroadcastDim)
    └─ 封装为统一元数据结构
    ↓
ElementwiseInfo 对象
    ├─ _meta[]: 紧凑存储所有元数据
    ├─ _output_size: 输出张量元素总数
    ├─ _input_size: 输入张量数量
    ├─ _ndim: 张量维度数
    └─ _output_contiguous: 输出是否连续存储
```

### 3.2 算子创建流程

```
用户调用: ElementwiseXXX_create()
    ↓
选择硬件后端(根据 device_type)
    ↓
 ELEMENTWISE_DESCRIPTOR 宏定义的 Descriptor::create()
    ├─ 调用 ElementwiseInfo::create() 生成元数据
    ├─ 调用 NAMESPACE::DeviceImpl::create() 初始化硬件
    └─ 计算 workspace_size(部分硬件需要额外工作空间)
    ↓
返回 infiniopDescriptor_t (统一句柄)
```

### 3.3 计算执行流程

```
用户调用: ElementwiseXXX_calculate()
    ↓
Descriptor::calculate()
    ├─ 传入 workspace、output、inputs、stream
    ├─ 从 _device_info 获取硬件特定实现
    └─ 调用 DeviceImpl::calculate() 执行实际计算
    ↓
硬件后端并行执行
    ├─ CPU: 串行/多线程遍历元素
    ├─ CUDA: GPU kernel 并行计算
    ├─ BANG: MLU 并行计算
    ├─ MUSA: 摩尔线程 GPU 并行
    └─ 其他国产加速卡: 各自的并行实现
```

### 3.4 广播机制处理

```
输入张量检查 (ElementwiseInfo::create)
    ↓
for each input:
    ├─ if input.ndim < output.ndim:
    │   └─ 标记为广播张量 (broadcasted=true)
    ├─ if input.hasBroadcastDim():
    │   └─ 标记为广播张量 (broadcasted=true)
    └─ 记录 input_contiguous 标志
    ↓
传递给硬件后端
    ↓
内核根据 broadcasted[] 标志
    ├─ 处理步长为 0 的广播维度
    └─ 实现自动索引扩展
```

### 3.5 模块依赖关系

```
elementwise.h (核心抽象层)
    ├─ 依赖: ../../utils.h (工具函数)
    ├─ 依赖: ../operator.h (算子基类)
    ├─ 依赖: ../tensor.h (张量描述符)
    └─ 被所有后端包含

各硬件后端
    ├─ bang/
    │   ├─ elementwise_bang_api.h (对外 API)
    │   ├─ elementwise_bang.h (设备实现)
    │   └─ elementwise_bang_kernel.h (内核代码)
    ├─ cpu/
    │   └─ elementwise_cpu.h (完整实现)
    ├─ kunlun/
    │   ├─ elementwise_kunlun_api.h
    │   └─ elementwise_kunlun.h
    ├─ metax/
    │   ├─ elementwise_metax_api.h
    │   └─ elementwise_metax.h
    ├─ moore/
    │   ├─ elementwise_moore_api.h
    │   └─ elementwise_moore.h
    └─ nvidia/
        ├─ elementwise_nvidia_api.cuh
        └─ elementwise_nvidia.cuh
```

### 3.6 设计模式应用

1. **策略模式**: 不同硬件后端实现各自的 DeviceImpl 策略
2. **工厂模式**: Descriptor::create() 根据硬件类型创建具体实现
3. **RAII**: ElementwiseInfo 使用移动语义管理元数据内存
4. **模板方法**: ELEMENTWISE_DESCRIPTOR 宏生成统一的算子骨架

## 4. 关键技术特性

- **零拷贝元数据**: ElementwiseInfo 使用单一 vector 紧凑存储所有元数据,减少内存分配
- **类型安全**: 强类型封装避免 C 风格指针错误
- **异常安全**: 使用 Result<T> 模式返回错误状态,避免异常开销
- **编译期多态**: 通过宏和模板实现编译期硬件分发,零运行时开销
- **内存对齐**: 使用 CEIL_DIV 确保 size_t 对齐,优化内存访问

## 5. 硬件支持矩阵

| 硬件厂商 | 后端目录 | 文件扩展名 | 编程语言 |
|---------|---------|-----------|---------|
| CPU     | cpu     | .h        | C++     |
| NVIDIA  | nvidia  | .cuh      | CUDA    |
| 寒武纪  | bang    | .h        | BangC   |
| 昆仑    | kunlun  | .h        | 自定义   |
| 天数智芯| metax   | .h        | 自定义   |
| 摩尔线程| moore   | .h        | MUSA    |
