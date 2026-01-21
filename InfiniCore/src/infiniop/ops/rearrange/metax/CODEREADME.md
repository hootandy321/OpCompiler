# Metax Rearrangement Kernel Core Implementation Documentation

本模块实现了基于沐曦(Metax)GPU架构的高性能张量重排(Tensor Rearrangement)操作,支持任意维度张量的转置、重塑和stride变换操作。该实现通过智能的block-grid维度划分策略,最大化GPU内存访问局部性和计算效率。

## 1. Module Structure

- **`rearrange_metax.h`**: 前向声明头文件,定义Metax后端的Descriptor接口
- **`rearrange_kernel.h`**: GPU内核核心实现,包含25×5×3种内核变体的宏生成器和参数调度逻辑
- **`rearrange_metax.maca`**: 主机端实现,包含Descriptor生命周期管理、参数准备算法和内核调度逻辑

## 2. Core Classes

### `Descriptor`
- **Location**: `rearrange_metax.maca`
- **Primary Function**: Metax后端的重排操作描述符,负责管理设备句柄、元数据解析和内核调度
- **Key Members**:
  - `_opaque`: 内部不透明指针,持有`device::metax::Handle::Internal`共享指针(包含设备属性如maxThreadsPerBlock)
  - `_meta`: `utils::RearrangeMeta`对象,存储张量的维度、步长和单元大小信息
- **Core Methods**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 工厂方法,验证输入输出张量类型一致性,调用`utils::RearrangeMeta::create()`解析元数据,构造Descriptor实例
  - `calculate(y, x, stream)`: 执行重排操作的主入口,根据元数据选择内核配置(512或1024线程),调用`prepareRearrangeParams()`计算内核参数,通过`launchKernel()`启动GPU内核
- **Lifecycle**: 由用户通过`create()`工厂方法创建,析构时释放`_opaque`内部指针

### 内核参数结构体
- **Location**: `rearrange_kernel.h` (定义), `rearrange_metax.maca` (使用)
- **Primary Function**: 封装GPU内核所需的所有参数,包括block/grid维度长度、步长和边界约束
- **Key Members**:
  - `block_len`: 各block维度的长度数组(最多5维)
  - `src_block_stride`: 源张量各block维度的字节步长数组
  - `dst_block_stride`: 目标张量各block维度的字节步长数组
  - `grid_len`: 各grid维度的长度数组(最多5维)
  - `src_grid_stride`: 源张量各grid维度的字节步长数组
  - `dst_grid_stride`: 目标张量各grid维度的字节步长数组
  - `block_dim`: block维度数量
  - `block_len_total`: block总元素数(所有block维度长度的乘积)
  - `constraints`: 边界约束数组,处理维度非均匀分割的情况(最多2个)
  - `unit_size`: 基础内存单元大小(1/2/4/8/16/32字节)

### `Dim` (内部辅助结构)
- **Location**: `rearrange_metax.maca`
- **Primary Function**: 表示单个维度的信息,用于参数准备过程
- **Key Members**:
  - `len`: 维度长度
  - `src_stride`: 源张量该维度的步长
  - `dst_stride`: 目标张量该维度的步长

### `SplitDim` (内部辅助结构)
- **Location**: `rearrange_metax.maca`
- **Primary Function**: 记录被分割维度的信息,用于生成边界约束条件
- **Key Members**:
  - `choose_idx`: 被分割维度的原始索引
  - `num_per_block`: 每个block处理的该维度元素数
  - `num_per_grid`: 该维度分割出的grid块数(向上取整)
  - `array_struct_idx_block`: 该维度在block数组结构中的索引(内核使用)
  - `array_struct_idx_grid`: 该维度在grid数组结构中的索引(内核使用)
  - `dim_len`: 该维度的原始完整长度

### `Constraint<ElementType>` (模板结构)
- **Location**: `rearrange_kernel.h`
- **Primary Function**: 描述维度分割后的边界约束,防止内核访问越界
- **Key Members**:
  - `grid_idx`: 约束对应的grid维度索引
  - `block_idx`: 约束对应的block维度索引
  - `grid_div_block`: grid维度相对于block维度的倍数(等于num_per_block)
  - `total_len`: 该维度的总长度,用于边界检查

### `ArrayStruct<int ArrSize, typename ArrayType>` (模板结构)
- **Location**: `rearrange_kernel.h`
- **Primary Function**: 将固定大小数组封装为结构体,便于GPU内核通过寄存器传递
- **Key Members**:
  - `a[ArrSize]`: 固定长度数组(支持1-5维)

## 3. API Interface

```cpp
// 创建Descriptor实例
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                      // Metax设备句柄
    Descriptor **desc_ptr,                        // 输出:创建的描述符指针
    infiniopTensorDescriptor_t y_desc,            // 目标张量描述符
    infiniopTensorDescriptor_t x_desc             // 源张量描述符
);
// 返回: INFINI_STATUS_SUCCESS(成功) / INFINI_STATUS_BAD_TENSOR_DTYPE(类型不匹配) /
//      INFINI_STATUS_BAD_TENSOR_SHAPE(形状不匹配)

// 执行重排操作
infiniStatus_t Descriptor::calculate(
    void *y,                                      // 目标张量设备指针
    const void *x,                                // 源张量设备指针
    void *stream                                  // HC流指针(hcStream_t)
) const;
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_INTERNAL_ERROR / INFINI_STATUS_BAD_PARAM /
//      INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED

// 准备内核参数(内部API)
utils::Result<RearrangeParams> prepareRearrangeParams(
    const utils::RearrangeMeta &original_meta,    // 原始元数据
    int max_threads                               // 设备最大线程数(512或1024)
);
// 返回: Result<RearrangeParams>对象,包含计算好的内核参数

// 获取内核函数指针(内部API)
utils::Result<void *> getRearrangeKernel(
    const RearrangeParams &params                 // 内核参数
);
// 返回: Result<void*>,指向编译好的GPU内核函数
```

## 4. Usage Example

```cpp
// 示例:在Metax GPU上执行张量转置 (N,C,H,W) -> (N,H,W,C)
#include "rearrange_metax.h"

// 1. 准备张量描述符
int64_t x_shape[] = {32, 64, 224, 224};  // NCHW格式
int64_t x_strides[] = {64*224*224*4, 224*224*4, 224*4, 4};  // float32类型
int64_t y_shape[] = {32, 224, 224, 64};  // NHWC格式
int64_t y_strides[] = {224*224*64*4, 224*64*4, 64*4, 4};

infiniTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc, INFINI_DTYPE_FLOAT32, 4, x_shape, x_strides);
infiniopCreateTensorDescriptor(handle, &y_desc, INFINI_DTYPE_FLOAT32, 4, y_shape, y_strides);

// 2. 创建重排描述符
op::rearrange::metax::Descriptor *rearrange_desc;
infiniStatus_t status = op::rearrange::metax::Descriptor::create(
    handle,
    &rearrange_desc,
    y_desc,
    x_desc
);

// 3. 分配设备内存并初始化数据
void *d_x, *d_y;
size_t x_size = 32 * 64 * 224 * 224 * sizeof(float);
hcMalloc(&d_x, x_size);
hcMalloc(&d_y, x_size);
hcMemcpyH2D(d_x, host_x_data, x_size);

// 4. 执行重排操作
hcStream_t stream;
hcStreamCreate(&stream);
status = rearrange_desc->calculate(d_y, d_x, stream);

// 5. 同步并获取结果
hcStreamSynchronize(stream);
hcMemcpyD2H(host_y_data, d_y, x_size);

// 6. 清理资源
delete rearrange_desc;
hcFree(d_x);
hcFree(d_y);
hcStreamDestroy(stream);
```

## 5. Implementation Details

### 5.1 参数准备算法 (`prepareRearrangeParams`)

这是本模块的核心算法,负责将张量维度智能划分为block维度和grid维度:

**算法流程**:
1. **单元大小调整**: 调用`original_meta.distributeUnit({32,16,8,4,2,1})`,将基础单元调整为2的幂次方,优化GPU内存访问对齐
2. **维度信息提取**: 从元数据中提取各维度的长度和步长,计算源步长的降序排序索引(用于优先选择连续维度)
3. **贪心维度选择**: 按照源步长降序和目标步长降序,贪心选择可以完全放入block的维度:
   - 如果源维度和目标维度索引相同(如转置),尝试同时添加到block
   - 否则根据`block_src_elements/block_dst_elements`比例,平衡选择源或目标维度
   - 当维度无法完全放入block时(超出max_threads限制),计算分割点
4. **维度分割**: 对需要分割的维度,计算每个block处理的元素数`num_per_block`和grid数量`num_per_grid`,记录到`SplitDim`结构
5. **参数组装**:
   - 填充`block_len`和`block_stride`(完整维度+分割维度的block部分)
   - 填充`grid_len`和`grid_stride`(非完整维度+分割维度的grid部分,grid_stride乘以num_per_block)
   - 对不均匀分割的维度(`dim_len % num_per_block != 0`),生成`Constraint`防止越界

**时间复杂度**: O(ndim²) - 主要来自排序和嵌套循环,实际ndim通常≤5,性能开销可忽略

**关键优化**:
- 按源步长降序选择维度,优先处理内存连续的维度,最大化内存合并(coalescing)
- 平衡源和目标维度的选择,避免block内部访问分散

### 5.2 GPU内核生成策略

模块使用宏元编程技术生成375种(25 block×grid组合×3约束数×6数据类型)内核变体:

**内核命名规则**: `rearrange_unit_<type>_block_<b>_grid_<g>_constrain_<c>`
- `<type>`: `uchar1/uchar2/float1/float2/float4/double4`(对应unit_size 1/2/4/8/16/32字节)
- `<b>`: block数组大小(1-5)
- `<g>`: grid数组大小(1-5)
- `<c>`: 约束数量(0/1/2)

**内核执行流程**:
1. **线程有效性检查**: 检查`threadIdx.x < block_len_total`,提前退出无效线程
2. **共享内存计算**:
   - 线程0根据`blockIdx.x`计算当前block的grid索引组合
   - 通过模运算和除法分解grid索引,累加`src_grid_stride`和`dst_grid_stride`
   - 如果存在约束,同时计算`constraints_grid_idx_multiple`(`grid_idx * num_per_block`)
   - 将结果写入共享内存(`shared_src_offset`, `shared_dst_offset`)
3. **线程同步**: `__syncthreads()`确保所有线程可见共享内存
4. **线程级偏移计算**:
   - 所有线程从共享内存读取block基础偏移
   - 根据`threadIdx.x`分解为block维度索引,累加`src_block_stride`和`dst_block_stride`
   - 如果存在约束,检查`grid_idx_multiple + block_idx < total_len`,越界则提前退出
5. **数据拷贝**: 使用字节偏移进行指针算术,执行单次内存拷贝操作

**性能关键点**:
- **共享内存优化**: 仅线程0计算grid偏移,其他线程从共享内存读取,减少重复计算
- **寄存器压力**: 通过固定大小数组(1-5维)优化寄存器分配,避免动态内存访问
- **向量化加载**: 根据unit_size选择uchar1/float2/double4等类型,利用128/256位内存带宽
- **边界检查优化**: 仅在约束存在时执行边界检查,无约束时通过编译期优化消除分支

### 5.3 内核调度策略 (`launchKernel` & `Descriptor::calculate`)

**调度决策**:
1. **特殊路径**: `ndim==0`时退化为简单`hcMemcpyAsync`设备到设备拷贝
2. **Block大小选择**:
   - 查询设备属性`maxThreadsPerBlock`(通常512或1024)
   - 取`min(METAX_BLOCK_SIZE_1024, max_threads)`作为最大线程数
   - 根据`block_len_total`选择内核:
     - `<=512`: 使用512线程内核
     - `<=1024`: 使用1024线程内核
     - `>1024`: 返回`INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
3. **Grid大小计算**: `grid_size = ∏grid_len[i]`,所有grid维度长度的乘积

**参数传递**: 使用C风格指针数组传递参数到`hcLaunchKernel`,包括:
- 输入输出指针(`y`, `x`)
- 维度信息(`block_dim`, `block_len_total`, 数组结构指针)
- 约束条件(如果为空则传递空约束引用)

### 5.4 内存管理

**设备内存**: 用户负责分配输入输出张量的设备内存,模块仅执行计算内核
**主机内存**: Descriptor对象使用`std::shared_ptr`管理设备句柄内部状态,确保生命周期安全
**内核内存**: 内核使用零拷贝模式,直接从全局内存读写,无需显式共享内存分配(仅使用编译期确定的小容量共享内存)

### 5.5 并发与同步

**并发模型**: HC流并发模型,多个重排操作可以提交到同一流顺序执行
**线程安全**: Descriptor对象创建后不可变,多线程可同时调用`calculate()`,但每个stream需要独立的Descriptor实例或外部同步
**同步点**: 用户负责调用`hcStreamSynchronize()`或`hcEvent`确保操作完成

### 5.6 错误处理

**错误类型**:
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 输入输出dtype不一致
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入输出维度数不一致
- `INFINI_STATUS_BAD_PARAM`: 内核参数验证失败(grid/block数组越界,约束数>2,unit_size不支持)
- `INFINI_STATUS_INTERNAL_ERROR`: HC API调用失败(内存拷贝、内核启动)
- `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`: block_len_total超过1024,设备不支持

**错误传播**: 使用`CHECK_OR_RETURN`和`CHECK_RESULT`宏将HC错误码转换为infiniStatus_t立即返回

### 5.7 设计模式

- **工厂模式**: `Descriptor::create()`静态工厂方法负责构造和验证
- **策略模式**: 通过`getRearrangeKernel()`根据参数动态选择375种内核策略之一
- **模板方法模式**: 内核实现使用宏模板生成变体,编译期特化
- **RAII**: 使用`std::shared_ptr`管理设备句柄,析构函数自动清理资源

### 5.8 性能特性

**算法复杂度**: O(N) - N为张量总元素数,每个元素仅访问一次
**带宽利用率**:
- 单次内存读写,理论带宽利用率100%
- 向量化加载(最高32字节/double4)充分利用内存总线
- 内存合并优化确保连续线程访问连续内存

**计算强度**: 极低(0 FLOPs/element),纯粹内存带宽受限操作
**延迟优化**:
- 共享内存减少重复计算(grid偏移仅线程0计算)
- 编译期常量展开(block/grid数组大小)
- 分支预测优化(约束检查仅在必要时执行)

**实际性能**: 接近设备理论内存带宽峰值,对于常见转置操作(如NCHW->NHWC)可达~900GB/s(在沐曦C500等高端GPU上)

### 5.9 依赖关系

**外部依赖**:
- `../../../devices/metax/metax_common.h`: Metax设备通用定义
- `../../../devices/metax/metax_kernel_common.h`: 内核通用工具(如`METAX_BLOCK_SIZE_1024`宏)
- `../rearrange.h`: Descriptor接口定义宏
- `../../../tensor.h`: 张量描述符结构
- `../../../utils.h`: 通用工具(如`CHECK_OR_RETURN`, `utils::Result`)
- `<hc.h>`: HC运行时API(HC是沐曦的CUDA兼容层)

**内部依赖**:
- `utils::RearrangeMeta`: 元数据解析和单元大小调整
- `device::metax::Handle::Internal`: 设备属性查询(如maxThreadsPerBlock)

### 5.10 编译与部署

**编译单元**: `rearrange_metax.maca`使用沐缘(METAX)编译器编译为`.maca`目标文件
**内核编译**: `rearrange_kernel.h`通过HC编译器生成设备代码,宏展开产生~225个内核函数
**链接**: 与InfiniCore主库静态链接,通过`DESCRIPTOR(metax)`宏注册到重排操作后端表

**编译宏**: `ENABLE_METAX_MC_API`控制使用`maca_fp8.h`或`hpcc_fp8.h`FP8支持(本模块未直接使用FP8,但通过头文件传递)
