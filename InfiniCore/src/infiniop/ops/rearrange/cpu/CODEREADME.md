# CPU Rearrange 算子核心实现文档

本模块实现了 Infini 框架中 CPU 后端的张量重排（rearrange）操作，通过 `utils::RearrangeMeta` 通用元数据优化和张量步长归约算法，实现高效的多线程内存重排，支持 transpose、reshape、permute 等多种张量变换场景。

## 1. 模块结构

- **`rearrange_cpu.h`**: CPU 后端描述符头文件，通过宏 `DESCRIPTOR(cpu)` 展开 `op::rearrange::cpu::Descriptor` 类定义
- **`rearrange_cpu.cc`**: CPU 后端实现，包含描述符创建、参数验证和 `RearrangeMeta::launch` 调用

## 2. 核心类

### `op::rearrange::cpu::Descriptor`
- **位置**: `rearrange_cpu.h` (宏展开), `rearrange_cpu.cc` (实现)
- **主要功能**: 封装 CPU 设备上张量重排操作的元数据和设备信息，提供类型安全的接口
- **关键成员**:
  - `_meta`: `utils::RearrangeMeta` 类型，存储张量的维度长度、索引步长、源步长、目标步长、单元大小和元素总数
  - `_opaque`: `Opaque*` 类型，指向设备特定的不透明数据结构（CPU 实现为 `nullptr`）
  - 继承自 `InfiniopDescriptor`，包含 `device_type` 和 `device_id`
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证输入/输出张量的数据类型、维度和形状一致性，调用 `RearrangeMeta::create` 生成优化后的元数据，构造描述符实例
  - `calculate(y, x, stream)`: 执行张量重排计算，调用 `_meta.launch(y, x)` 触发多线程内存拷贝（忽略 `stream` 参数）
  - `~Descriptor()`: 析构函数（默认实现）
- **生命周期**: 由用户通过 `create` 方法构造并分配在堆上，使用完毕后通过 `infiniopDestroyRearrangeDescriptor` 删除

### `utils::RearrangeMeta` (核心辅助类)
- **位置**: `/home/qy/src/Infini/InfiniCore/src/utils/rearrange.h` (接口), `/home/qy/src/Infini/InfiniCore/src/utils/rearrange.cc` (实现)
- **主要功能**: 封装张量重排的优化元数据，通过维度合并和步长归约减少循环嵌套层级，支持 OpenMP 并行化
- **关键成员**:
  - `_meta`: `std::vector<ptrdiff_t>` 类型，紧凑的元数据布局，结构为 `[unit, idx_strides[ndim], dst_strides[ndim], src_strides[ndim]]`（长度 `2 + ndim * 3`）
- **核心方法**:
  - `create(shape, dst_strides, src_strides, ndim, element_size)`: 静态工厂方法，执行以下优化步骤：
    1. 剔除长度为 1 的维度
    2. 按目标步长绝对值降序排序（最大化内存连续性）
    3. 合并末尾连续维度到 `unit`（如果步长等于 `unit`）
    4. 合并任意连续维度（满足 `stride[i] * len[i+1] == stride[i-1]`）
    5. 计算索引步长（维度累积乘积）
  - `launch(dst, src)`: 执行重排操作，如果 `count == 1` 则调用 `std::memcpy`，否则使用 OpenMP 并行 for 循环遍历所有元素，通过除法和取余计算多维索引
  - `distributeUnit(candidates)`: 将 `unit` 拆分为更小的 2 的幂次方（如 `{32, 16, 8, 4, 2, 1}`），用于 GPU 后端优化并行粒度
  - `ndim()`, `unit()`, `count()`: 访问器方法，返回优化后的维度数、单元大小和元素总数
  - `idx_strides()`, `dst_strides()`, `src_strides()`: 返回内部元数据指针
- **算法复杂度**:
  - `create`: O(ndim log ndim) 时间（排序）+ O(ndim) 空间
  - `launch`: O(count * ndim) 时间，O(count) 空间（OpenMP 线程栈）

### `InfiniopTensorDescriptor` (张量描述符)
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/tensor.h`
- **主要功能**: 封装张量的数据类型、形状和步长信息
- **关键方法**:
  - `dtype()`, `ndim()`, `shape()`, `strides()`: 访问器方法
  - `isMergable(dim_start, dim_end)`: 检查维度区间是否可以合并
  - `isContiguous()`: 检查张量是否内存连续
  - `hasBroadcastDim()`: 检查是否存在广播维度（步长为 0 但维度长度 > 1）

### `device::cpu::Handle` (设备句柄)
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/cpu/cpu_handle.h`
- **主要功能**: 封装 CPU 设备信息（继承自 `InfiniopHandle`，包含 `device` 和 `device_id`）
- **创建方法**: `create(handle_ptr, device_id)` 静态工厂方法

## 3. API 接口

```cpp
// 创建 Rearrange 描述符（C API）
infiniStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,                      // Infini 操作句柄
    infiniopRearrangeDescriptor_t *desc_ptr,      // 输出：描述符指针
    infiniopTensorDescriptor_t dst,               // 输出张量描述符
    infiniopTensorDescriptor_t src                // 输入张量描述符
);
// 返回 INFINI_STATUS_SUCCESS 或错误码（dtype/shape/strides 不匹配）

// 执行张量重排计算（C API）
infiniStatus_t infiniopRearrange(
    infiniopRearrangeDescriptor_t desc,           // 描述符
    void *dst,                                    // 输出缓冲区（主机内存）
    const void *src,                              // 输入缓冲区（主机内存）
    void *stream                                  // 流参数（CPU 实现忽略）
);
// 返回 INFINI_STATUS_SUCCESS

// 销毁描述符（C API）
infiniStatus_t infiniopDestroyRearrangeDescriptor(
    infiniopRearrangeDescriptor_t desc            // 描述符
);
```

## 4. 使用示例

```cpp
// 示例：在 CPU 上执行张量转置 (2, 3) -> (3, 2)

#include "infiniop.h"
#include "infiniop/operator_descriptor.h"
#include "infiniop/tensor_descriptor.h"

// 1. 准备张量描述符
std::vector<size_t> x_shape = {2, 3};
std::vector<ptrdiff_t> x_strides = {3, 1};  // 行主序（C 风格）
std::vector<size_t> y_shape = {3, 2};
std::vector<ptrdiff_t> y_strides = {1, 2};  // 转置后的步长

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(
    &x_desc,
    INFINI_DEVICE_CPU, 0,
    INFINI_DTYPE_F32,
    x_shape.size(), x_shape.data(), x_strides.data()
);
infiniopCreateTensorDescriptor(
    &y_desc,
    INFINI_DEVICE_CPU, 0,
    INFINI_DTYPE_F32,
    y_shape.size(), y_shape.data(), y_strides.data()
);

// 2. 创建 CPU 句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

// 3. 创建 Rearrange 描述符
infiniopRearrangeDescriptor_t rearrange_desc;
auto status = infiniopCreateRearrangeDescriptor(
    handle, &rearrange_desc, y_desc, x_desc
);
// 内部调用流程：
// - Descriptor::create() 验证 dtype == INFINI_DTYPE_F32
// - 验证 ndim == 2
// - 验证 shape 一致性（flatten 后元素数相同）
// - 调用 RearrangeMeta::create({2, 3}, {1, 2}, {3, 1}, 2, 4)
//   - 优化结果：unit = 4, ndim = 2, count = 6
//   - idx_strides = {3, 1}, dst_strides = {4, 8}, src_strides = {12, 4}

// 4. 准备数据
std::vector<float> x = {1, 2, 3, 4, 5, 6};  // shape (2, 3), 行主序
std::vector<float> y(6);                    // shape (3, 2)

// 5. 执行重排
status = infiniopRearrange(
    rearrange_desc,
    y.data(),  // 输出缓冲区
    x.data(),  // 输入缓冲区
    nullptr    // stream（CPU 忽略）
);
// 内部调用流程：
// - Descriptor::calculate() 调用 _meta.launch(y.data(), x.data())
// - RearrangeMeta::launch() 启动 OpenMP 并行循环：
//   for (int i = 0; i < 6; ++i) {
//       int rem = i;
//       for (int j = 0; j < 2; ++j) {
//           int k = rem / idx_strides[j];  // j=0: k=i/3, j=1: k=i%3
//           dst += k * dst_strides[j];
//           src += k * src_strides[j];
//           rem %= idx_strides[j];
//       }
//       memcpy(dst, src, 4);  // unit=4 字节
//   }
// - 结果：y = {1, 4, 2, 5, 3, 6}（转置）

// 6. 清理资源
infiniopDestroyRearrangeDescriptor(rearrange_desc);
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **元数据存储**: `RearrangeMeta` 使用 `std::vector<ptrdiff_t>` 紧凑存储所有元数据，避免堆分配碎片
- **零拷贝优化**: 当输入/输出张量内存布局完全一致时，`RearrangeMeta::launch` 检测到 `count == 1`，直接调用 `std::memcpy`，避免循环开销
- **设备无关**: CPU 实现无需管理设备内存，所有操作在主机内存执行

### 并发策略
- **OpenMP 并行化**: `RearrangeMeta::launch` 使用 `#pragma omp parallel for` 并行化外层循环（元素级别），适用于多核 CPU
- **数据局部性**: 通过步长归约和维度合并最大化内存连续访问，提高缓存命中率
- **无锁设计**: 每个线程处理独立的输出元素，无需同步机制

### 性能优化
- **维度合并算法**: `RearrangeMeta::create` 自动合并满足条件的连续维度，减少循环嵌套层级和索引计算开销
  - **末尾合并**: 如果维度步长等于当前 `unit`，将该维度长度合并到 `unit`（例如 `element_size=4`, `stride=4` -> `unit=4*len`）
  - **任意合并**: 如果满足 `stride[i] * len[i+1] == stride[i-1]`，合并两个维度为一个
- **步长排序**: 按目标步长绝对值降序排序，确保最外层循环对应最大步长，提高内存访问连续性
- **单元大小优化**: `unit` 存储每次 `memcpy` 的字节数，减少小数据拷贝次数（例如拷贝 float4 而非 float）

### 错误处理
- **参数验证**:
  - `Descriptor::create` 检查输入/输出张量的 `dtype` 一致性（返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`）
  - 检查 `ndim` 一致性（返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - 检查 `shape` 一致性（使用 `CHECK_SAME_SHAPE` 宏，返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - `RearrangeMeta::create` 检查目标步长不为 0（返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`）
- **Result 类型**: `RearrangeMeta::create` 返回 `utils::Result<RearrangeMeta>`，封装成功值或错误码，使用 `CHECK_RESULT` 宏检查错误
- **调试信息**: `check.h` 中的宏在断言失败时输出错误位置（`__func__`, `__FILE__`, `__LINE__`）

### 依赖关系
- **外部依赖**:
  - `infinicore.h`: 提供 `infiniDtype_t`, `infiniDevice_t`, `infiniStatus_t` 等基础类型
  - OpenMP（可选）: 通过 `#ifdef ENABLE_OMP` 启用多线程支持
- **内部模块依赖**:
  - `utils::RearrangeMeta`: 核心元数据管理和重排逻辑（`/home/qy/src/Infini/InfiniCore/src/utils/rearrange.cc`）
  - `utils::Result<T>`: 错误处理封装（`/home/qy/src/Infini/InfiniCore/src/utils/result.hpp`）
  - `CHECK_OR_RETURN`, `CHECK_SAME_SHAPE`, `CHECK_RESULT`: 参数验证宏（`/home/qy/src/Infini/InfiniCore/src/utils/check.h`）
  - `infiniSizeOf()`: 计算 dtype 的字节大小（`/home/qy/src/Infini/InfiniCore/src/utils.h`）

### 设计模式
- **策略模式**: `rearrange_cpu.h` 通过宏 `DESCRIPTOR(cpu)` 展开生成 `op::rearrange::cpu::Descriptor` 类，避免代码重复（多个硬件后端共享同一套接口定义）
- **工厂方法**: `Descriptor::create` 作为静态工厂方法封装对象创建逻辑
- **RAII**: 描述符对象由用户管理生命周期，析构函数自动清理 `_opaque` 资源
- **零开销抽象**: `RearrangeMeta` 使用紧凑的 `std::vector<ptrdiff_t>` 布局，运行时无虚表开销

### 算法细节
- **索引计算**: `RearrangeMeta::launch` 使用除法和取余算法计算多维索引：
  ```cpp
  auto rem = i;  // 线性索引
  for (size_t j = 0; j < ndim_; ++j) {
      auto k = rem / idx_strides_[j];  // 当前维度索引
      dst += k * dst_strides_[j];      // 累加目标偏移
      src += k * src_strides_[j];      // 累加源偏移
      rem %= idx_strides_[j];          // 更新剩余索引
  }
  ```
  - `idx_strides` 是维度累积乘积（例如 shape `{2, 3, 4}` -> `idx_strides {12, 4, 1}`）
  - 通过除法和取余避免递归或栈分配索引数组
- **步长归约**: `RearrangeMeta::create` 将步长从元素单位转换为字节单位（`stride * unit`），统一内存偏移计算

### 与其他后端的对比
- **NVIDIA/Moore/MetaX 后端**:
  - 使用 `prepareRearrangeParams` 将 `RearrangeMeta` 转换为 GPU 特定的 `RearrangeParams` 结构
  - 调用 `distributeUnit({32, 16, 8, 4, 2, 1})` 拆分单元大小以优化 block 并行度
  - 启动特化的 CUDA Kernel（225 个模板实例）而非 OpenMP 循环
- **CPU 后端**:
  - 直接使用 `RearrangeMeta::launch`，无需额外参数转换
  - 依赖 OpenMP 自动并行化，无需手动配置线程块
  - 适合中小规模张量和主机端计算场景
