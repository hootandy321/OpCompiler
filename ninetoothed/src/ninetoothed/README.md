# `ninetoothed` 符号化张量编译器模块文档

## 模块概述

**ninetoothed** 是一个基于符号计算的张量编程系统,用于自动生成高性能GPU计算内核。它通过符号化的张量操作和代码生成技术,将类Python的描述性代码转换为优化的Triton/CUDA内核,同时支持JIT(即时编译)和AOT(提前编译)两种编译模式。

该模块的核心价值在于:
- **符号化张量抽象**: 通过`Tensor`和`Symbol`类实现符号化的张量操作,支持运行时形状推导
- **自动化代码生成**: 将Python AST转换为优化的GPU内核代码(基于Triton)
- **层次化内存布局**: 支持分块(tiling)、膨胀(expansion)等高级内存布局优化
- **多后端支持**: 可生成PyTorch调用代码或纯CUDA C代码

## 模块结构 (Module Structure)

### 核心类与组件

- **`Symbol`** (`symbol.py`): 符号表达式类,基于Python AST实现符号运算
- **`Tensor`** (`tensor.py`): 符号化张量类,支持层次化内存布局
- **`CodeGenerator`** (`generation.py`): AST代码生成器,将Python函数转换为Triton内核
- **`JIT`** (`jit.py`): JIT编译接口,提供`@jit`装饰器
- **`aot`** (`aot.py`): AOT编译接口,生成C/CUDA源文件

### 辅助工具模块

- **`dtype.py`**: 数据类型定义(i8, i32, fp16, fp32等)
- **`language.py`**: 语言转换层,将ninetoothed DSL映射到Triton语言
- **`naming.py`**: 命名管理,处理符号前缀(constexpr, meta等)
- **`cudaifier.py`**: AST转换器,将符号转换为CUDA C代码
- **`torchifier.py`**: AST转换器,将符号转换为PyTorch调用
- **`eval.py`**: 符号求值器,将符号张量求值为数值张量
- **`debugging.py`**: 调试工具,提供张量布局模拟
- **`visualization.py`**: 可视化工具,生成张量内存布局图

### 构建与接口

- **`make.py`**: 统一接口,整合arrangement和application
- **`build.py`**: 批量构建接口,支持多配置内核生成
- **`utils.py`**: 工具函数,计算默认配置参数

## 核心类详解 (Key Classes)

### 1. `Symbol` 类

**定义位置**: `symbol.py:9-262`

**主要功能**:
表示符号表达式,支持运算符重载和AST操作。符号可用于表达张量形状、索引、偏移量等编译时不确定的值。

**关键成员**:
- `_node`: Python AST节点,存储表达式的抽象语法树
- `lower_bound`: 符号取值范围下界(用于自动调优)
- `upper_bound`: 符号取值范围上界
- `power_of_two`: 是否为2的幂次元参数

**核心方法**:
- `__add__/__sub__/__mul__/__floordiv__`: 运算符重载,返回符号化二元运算结果
- `find_and_replace(target, replacement)`: 在符号树中查找并替换子表达式
- `names()`: 收集符号树中所有引用的符号变量

**符号类型**:
1. **普通符号**: `Symbol("n")` - 表示变量n
2. **常量表达式符号**: `Symbol("n", constexpr=True)` - 编译时常量
3. **元参数符号**: `block_size()` - 自动调优的块大小参数

**实现细节**:
```python
# Symbol基于Python AST实现
expr = Symbol("i + 1")  # 解析为AST BinOp节点
result = expr * 2        # 返回新的Symbol,AST为 BinOp(Mult, expr, 2)

# 符号前缀命名约定
constexpr符号: ninetoothed_constexpr_prefix_n
meta符号: ninetoothed_meta_prefix_n
```

### 2. `Tensor` 类

**定义位置**: `tensor.py:11-757`

**主要功能**:
表示符号化张量,支持复杂的内存布局变换(分块、膨胀、维度变换等)。张量可以形成层次化结构,每个层级可以有独立的形状和索引映射。

**关键成员**:
- `shape`: 张量形状符号元组 `(n0, n1, ...)`
- `dtype`: 数据类型或嵌套的Tensor(用于层次化结构)
- `jagged_dim`: 变长维度(支持Jagged Tensor)
- `other`: 越界填充值
- `source`: 源张量引用,用于追溯内存布局
- `_levels`: 层次化层级列表
- `_offsets`: 索引映射函数,计算从当前张量到源张量的偏移

**核心方法**:

#### 内存布局变换
- `tile(tile_shape, strides, dilation, floor_mode)`: **核心方法** - 将张量分块为层次结构
  ```python
  # 示例: 将(N, C, H, W)张量分块为(N/tile_n, C/tile_c, tile_n, tile_c, H, W)
  tiled = tensor.tile((tile_n, tile_c, 1, 1))
  ```

- `expand(shape)`: 扩展单维度(类似NumPy的broadcast)
- `unsqueeze(dim)`: 插入单维度
- `squeeze(dim)`: 删除单维度
- `permute(dims)`: 维度重排
- `flatten(start_dim, end_dim)`: 展平连续维度
- `ravel()`: 展平所有层次(与flatten不同,ravel会展开嵌套的dtype结构)

#### 索引操作
- `__getitem__(indices)`: 支持完整索引语法
  ```python
  tensor[0, ...]          # 整数索引
  tensor[0:10, ::2]       # 切片索引
  tensor[None, :, None]   # 插入维度
  ```

#### 内部方法
- `_slice_dim(dim, start, stop, step)`: 沿指定维度切片
- `offsets()`: 计算索引到源张量的偏移量并更新mask

**命名约定**:
- 形状符号: `tensor_size_{dim}`
- 步幅符号: `tensor_stride_{dim}`
- 指针符号: `tensor_pointer`
- Jagged张量: `tensor_values`, `tensor_offsets`, `tensor_seq_len`

### 3. `CodeGenerator` 类

**定义位置**: `generation.py:28-830`

**主要功能**:
AST代码生成器,将ninetoothed函数转换为可执行的Triton内核。实现函数内联、自动调优配置生成、负载/存储代码生成。

**关键成员**:
- `_context`: 函数参数类型注解(张量列表)
- `_symbols`: 符号表,映射符号名到Symbol对象
- `_autotune`: 自动调优配置AST节点(如果生成)
- `_invariants`: 不变量集合(在函数体中不变的符号表达式)
- `_min_num_elements`: 最小元素数(根据GPU寄存器限制)
- `_max_num_elements`: 最大元素数(避免寄存器溢出)

**核心方法**:

#### AST遍历与转换
- `visit_FunctionDef(node)`: 处理函数定义
  1. 提取张量参数类型注解
  2. 生成不变量赋值语句
  3. 生成自动调优配置
  4. 添加`@triton.jit`装饰器
  5. 生成launch函数

- `visit_Call(node)`: 处理函数调用
  - `tensor.data_ptr()`: 转换为指针符号
  - `tensor.offsets(dim)`: 生成索引偏移表达式
  - `tensor.stride(dim)`: 转换为步幅符号

- `visit_Subscript(node)`: 处理张量索引 `tensor[i, j]`
  - 调用`_generate_load()`生成加载代码
  - 计算指针地址和边界mask

- `visit_Assign(node)`: 处理赋值 `target[i, j] = value`
  - 调用`_generate_store()`生成存储代码

#### 代码生成
- `_generate_load(tensor, indices)`: 生成triton.language.load调用
  ```python
  # 伪代码
  pointers = base_pointer + overall_offsets
  mask = in_bounds_check
  return triton.language.load(pointers, mask=mask, other=padding)
  ```

- `_generate_store(tensor, value, indices)`: 生成triton.language.store调用

- `_generate_pointers_and_mask(tensor, indices)`:
  - 计算整体偏移: `sum(offsets[dim] * strides[dim])`
  - 生成边界检查mask

- `_generate_autotune(params, meta)`: **关键方法** - 生成自动调优配置
  1. 使用SymPy求解元参数取值空间
  2. 生成`[triton.Config(...)]`配置列表
  3. 添加`@triton.autotune`装饰器

- `_generate_launch(params, meta)`: 生成kernel启动函数
  - 对于Torch: 生成torch调用接口
  - 对于CUDA: 生成C调用接口

**内联优化**:
`_Inliner`类实现函数内联,将小函数调用展开为内联代码,减少调用开销。

### 4. `JIT` 类

**定义位置**: `jit.py:61-129`

**主要功能**:
JIT编译接口,将Python函数编译为可调用的内核句柄。

**执行流程**:
1. 创建`CodeGenerator`实例
2. 生成Triton内核源代码
3. 缓存源代码文件(基于SHA256哈希)
4. 动态导入生成的模块
5. 返回`_Handle`对象,封装kernel和launch函数

**使用示例**:
```python
@ninetoothed.jit
def kernel(x):
    x[i] = x[i] * 2

kernel(tensor)  # 即时编译并执行
```

### 5. `aot` 函数

**定义位置**: `aot.py:15-140`

**主要功能**:
提前编译,生成C/CUDA源文件和头文件,供外部C/C++项目调用。

**编译流程**:
1. 生成Triton内核源码
2. 调用`triton.tools.compile`将Triton编译为PTX
3. 生成C包装函数,类型转换参数
4. 输出`.c`和`.h`文件

**生成的API**:
```c
// C接口
NineToothedResult launch_kernel(
    NineToothedStream stream,
    NineToothedTensor input,
    NineToothedTensor output
);
```

## ninetoothed API (Namespace API)

### 符号创建

#### `Symbol(expr, constexpr, meta, lower_bound, upper_bound, power_of_two)`
创建符号表达式。

**参数**:
- `expr`: 表达式(字符串/AST/CodeType/Symbol对象)
- `constexpr`: 是否为编译时常量
- `meta`: 是否为元参数(自动调优)
- `lower_bound/upper_bound`: 取值范围
- `power_of_two`: 是否为2的幂次

**返回**: Symbol对象

#### `block_size(lower_bound, upper_bound)`
创建块大小元参数,用于自动调优。

**默认范围**: [32, 1024],仅包含2的幂次

### 张量操作

#### `Tensor(ndim, shape, dtype, jagged_dim, other, shape_options, constexpr, value, name)`
创建符号化张量。

**参数**:
- `ndim`: 张量维度数
- `shape`: 形状(可选,自动生成符号)
- `dtype`: 数据类型(字符串,如"fp32")
- `jagged_dim`: 变长维度索引
- `other`: 越界填充值
- `shape_options`: 形状符号选项(如`{"constexpr": True}`)
- `constexpr`: 是否为常量张量
- `value`: 常量张量的值
- `name`: 张量名称(可选,自动生成)

#### `Tensor.tile(tile_shape, strides, dilation, floor_mode)`
分块操作,创建层次化张量。

**参数**:
- `tile_shape`: 每个块的形状(元组,-1表示全尺寸)
- `strides`: 块间步幅(元组,-1表示等于块大小)
- `dilation`: 块内膨胀(元组)
- `floor_mode`: 是否使用向下取整计算外层形状

**示例**:
```python
# 将(N, 64)张量每64行分为一块
tiled = Tensor((N, 64)).tile((1, 64))
# 结果: 外层形状(N, 1), 内层形状(1, 64)
```

### 编译接口

#### `jit(func, caller, kernel_name, num_warps, num_stages, max_num_configs)`
JIT编译装饰器。

**参数**:
- `func`: 被编译函数(或省略作为装饰器工厂)
- `caller`: 调用者类型("torch"或"cuda")
- `kernel_name`: 生成的内核名称
- `num_warps`: Warp数量(默认自动计算)
- `num_stages`: 流水线阶段数(默认自动计算)
- `max_num_configs`: 最大自动调优配置数

**返回**: 可调用句柄

#### `make(arrangement, application, tensors, caller, kernel_name, ...)`
统一编译接口,整合arrangement和application。

**arrangement**: 张量布局函数,返回张量类型注解
**application**: 计算内核函数

**示例**:
```python
def arrangement(x, y):
    return x, y  # 类型注解

@ninetoothed.make
def application(x, y):
    y[i] = x[i] * 2

kernel = make(arrangement, application, tensors)
```

#### `build(premake, configs, caller, kernel_name, output_dir)`
批量构建接口,生成多配置内核。

**configs**: 配置列表,每个配置为`(args, kwargs, compilation_configs)`

### 符号求值

#### `eval(tensor, subs)`
将符号张量求值为数值张量(NumPy数组)。

**subs**: 符号替换字典,将符号替换为具体值

**示例**:
```python
n = Symbol("n")
tensor = Tensor(2, shape=(n, 64))
result = eval(tensor, {n: 128})  # 返回shape=(128, 64)的数组
```

#### `subs(tensor, replacements)`
符号替换,返回新张量。

### 工具函数

#### `calculate_default_configs()`
根据GPU属性计算默认warp数和流水线阶段数。

**返回**: `(num_warps, num_stages)`

### 调试与可视化

#### `simulate_arrangement(arrangement, tensors, device)`
模拟张量布局变换,返回源张量和目标张量的索引映射。

**用途**: 验证内存布局正确性

#### `visualize(tensor, color, save_path)`
可视化张量内存布局为网格图。

#### `visualize_arrangement(arrangement, tensors)`
交互式可视化工具,显示源张量到目标张量的映射。

## 实现细节 (Implementation Details)

### 1. AST转换流水线

代码生成流程:
```
Python函数
    ↓ (解析AST)
函数内联 (_Inliner)
    ↓
符号替换 (Tritonizer - ninetoothed→triton)
    ↓
表达式简化 (_BinOpSimplifier)
    ↓
生成launch函数 (Torchifier/Cudaifier)
    ↓
缓存源文件 (SHA256哈希)
    ↓
动态导入 (importlib)
```

### 2. 自动调优算法

**元参数搜索空间生成** (`_generate_autotune`):
1. 对每个元参数`BLOCK_SIZE`,生成候选值集合
   - 如果`power_of_two=True`: 仅包含2的幂次
   - 否则: 包含范围内的所有整数
2. 使用SymPy简化不等式约束:
   ```python
   num_elements <= max_registers
   num_elements >= min_registers
   ```
3. 笛卡尔积生成所有配置组合
4. 生成`@triton.autotune`装饰器:
   ```python
   @triton.autotune(
       configs=[
           triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
           triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
           ...
       ],
       key=["M", "N", "K"]
   )
   ```

### 3. 层次化张量内存布局

**tile操作的内部表示**:
- **多级结构**: 每次tile创建新层级,`dtype`指向内层Tensor
- **索引计算**: 通过`_offsets`函数递归计算每层偏移
- **延迟求值**: 偏移量在实际使用时才计算(通过`offsets()`方法)

**示例**: `(N, 64)`张量tile为`(N/16, 16, 64)`
```
Level 0: 外层张量, shape=(N/16,), dtype指向Level 1
Level 1: 内层张量, shape=(16, 64), dtype=None(实际数据)
```

### 4. 边界检查与Mask生成

**Mask计算** (`_generate_overall_offsets_and_mask`):
```python
mask = True
for dim in range(ndim):
    mask &= (indices[dim] >= 0) & (indices[dim] < shape[dim])

# 对于Jagged张量
if jagged_dim is not None:
    seq_len = load(offsets[batch_id + 1]) - load(offsets[batch_id])
    mask &= (seq_index < seq_len)
```

**越界处理**: 通过`other`参数指定填充值(如`-inf`, `0`)

### 5. 命名约定系统

**前缀命名** (`naming.py`):
- 自动生成: `ninetoothed_{name}`
- constexpr: `ninetoothed_constexpr_prefix_{name}`
- meta: `ninetoothed_meta_prefix_{name}`
- next_power_of_2: `ninetoothed_next_power_of_2_prefix_{name}`

**用途**: 避免符号名冲突,区分不同类型的符号

### 6. 目标代码生成

**Torchifier转换**:
```python
# 符号 → PyTorch调用
tensor_pointer           → tensor
tensor_size_0            → tensor.shape[0]
tensor_stride_0          → tensor.stride(0)
tensor_values()          → tensor.values()
tensor_offsets()         → tensor.offsets()
```

**Cudaifier转换**:
```python
# 符号 → CUDA C结构体访问
tensor_pointer           → tensor.data
tensor_size_0            → tensor.shape[0]
tensor_stride_0          → tensor.strides[0]
```

### 7. 代码缓存机制

**缓存策略** (`cache_source`):
- 使用SHA256哈希源代码
- 缓存目录: `~/.ninetoothed/{hash}.py`
- 避免重复编译相同内核

### 8. JIT与AOT对比

| 特性 | JIT模式 | AOT模式 |
|------|---------|---------|
| 编译时机 | 运行时首次调用 | 构建时 |
| 输出 | Python模块 | C源文件+头文件 |
| 调用方式 | Python函数 | C函数指针 |
| 适用场景 | Python原型开发 | 生产环境部署 |

## 使用示例 (Usage Example)

### 示例1: 简单向量加法

```python
import ninetoothed as nt

@nt.jit
def vector_add(x, y):
    y[i] = x[i] + y[i]

import torch
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

vector_add(x, y)  # JIT编译并执行
```

### 示例2: 矩阵乘法(Tiling优化)

```python
@nt.jit
def matmul(a, b, c):
    # 分块优化
    a_tile = a.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_tile = b.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))

    for m in range(BLOCK_SIZE_M):
        for k in range(BLOCK_SIZE_K):
            for n in range(BLOCK_SIZE_N):
                c[m, n] += a_tile[m, k] * b_tile[k, n]
```

### 示例3: AOT编译为C代码

```python
def arrangement(x, y):
    return x, y

def kernel(x, y):
    y[i] = x[i] * 2

nt.aot(
    kernel,
    caller="cuda",
    kernel_name="scale",
    output_dir="./kernels"
)

# 生成文件:
# - scale.{hash}.c
# - scale.{hash}.h
```

### 示例4: 自动调优

```python
BLOCK_SIZE_M = nt.block_size(lower_bound=32, upper_bound=128)
BLOCK_SIZE_N = nt.block_size(lower_bound=32, upper_bound=128)

@nt.jit
def tuned_kernel(x, y):
    x_tile = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    # ... 计算逻辑

# 自动生成多个配置,运行时选择最优
```

## 技术特点

1. **符号计算**: 基于Python AST实现完整的符号表达式系统
2. **类型推导**: 通过Python类型注解实现张量类型推导
3. **零拷贝**: 符号操作仅为AST变换,无实际数据移动
4. **编译器技术**: 函数内联、常量折叠、死代码消除
5. **自动优化**: 根据GPU属性自动生成最优配置
6. **多级抽象**: 从符号→AST→Triton→PTX/CUDA的完整编译栈

## 依赖关系

- **Python 3.x**: AST处理、元编程
- **Triton**: GPU内核DSL和编译器
- **SymPy**: 符号数学(用于自动调优约束求解)
- **PyTorch**: (可选) JIT模式的张量后端
- **NumPy**: 符号求值和调试

## 性能考虑

1. **寄存器限制**: `_max_num_elements`根据GPU寄存器数量设置,避免溢出
2. **共享内存**: `num_stages`根据GPU共享内存大小设置
3. **Warp调度**: 默认使用8个warp(256线程)
4. **代码膨胀**: 过多自动调优配置会增加编译时间和二进制大小

## 限制与注意事项

1. 仅支持CUDA后端(NVIDIA GPU)
2. 需要Triton编译器工具链
3. 复杂控制流(如嵌套循环)可能需要手动优化
4. 符号表达式过于复杂时,AST遍历可能成为瓶颈
