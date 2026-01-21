# NineToothed 算子构建工具核心实现文档

NineToothed 算子构建工具是 InfiniOP 框架中用于自动化生成 CUDA kernel 代码的构建系统。该模块通过元编程技术，将 Python 函数转换为高度优化的 CUDA kernel，并生成配套的 C 语言接口封装，实现了从高层算子定义到底层 GPU 实现的全自动化构建流程。

## 1. 模块结构

- **`build.py`**: 核心构建引擎，负责遍历算子参数空间、调用 NineToothed 代码生成器、生成多版本的 kernel 实现及 C 语言接口封装代码

## 2. 核心函数

### `build(premake, constexpr_param_grid, caller, op_name, output_dir)`
- **位置**: `build.py` (第 16-98 行)
- **主要功能**: 根据参数网格自动生成算子的多版本 CUDA kernel 实现，并生成统一的 C 语言调度接口
- **参数**:
  - `premake`: 算子预构建函数，接受参数组合返回 `(arrangement, application, tensors)` 三元组
    - `arrangement`: 内存布局转换函数，定义输入输出张量的分块策略
    - `application`: 算子核心计算逻辑（Python 函数）
    - `tensors`: 张量元组，定义输入输出的形状和数据类型
  - `constexpr_param_grid`: 字典，键为编译期常量参数名，值为该参数所有可能的取值列表
  - `caller`: 目标后端标识符（如 `"cuda"`）
  - `op_name`: 算子名称，用于生成函数名和文件名
  - `output_dir`: 输出目录路径，生成的 `.c` 和 `.h` 文件将写入此目录
- **核心算法**:
  1. **笛卡尔积枚举** (第 21 行): 使用 `itertools.product` 计算所有参数组合的笛卡尔积，生成完整的参数空间遍历
  2. **dtype 标准化** (第 24-28 行): 将字符串形式的 dtype（如 `"fp16"`）转换为 InfiniCore 宏定义（如 `INFINI_DTYPE_F16`）
  3. **kernel 命名** (第 30-32 行): 为每个参数组合生成唯一 kernel 名，格式为 `{op_name}_{param1_value}_{param2_value}_...`
  4. **NineToothed 代码生成** (第 34-41 行): 调用 `ninetoothed.make()` 生成 CUDA kernel 源码
  5. **C 接口生成** (第 43-86 行): 生成包含条件分发逻辑的 C 包装函数
- **生成文件结构**:
  ```
  {output_dir}/{op_name}.c       # C 源文件，包含所有 kernel 的 #include 和调度函数
  {output_dir}/{op_name}.h       # C 头文件，声明函数接口
  {output_dir}/{op_name}_{param_hash}.h  # 每个 kernel 变体的头文件（由 ninetoothed.make 生成）
  ```

### `_generate_condition(combination)`
- **位置**: `build.py` (第 100-101 行)
- **功能**: 将参数字典转换为 C 语言条件表达式，用于运行时分发
- **算法**: 拼接 `param == value` 形式的条件子句，使用 `&&` 连接
- **示例**: `{"ndim_": 3, "dtype_": "INFINI_DTYPE_F32"}` → `"ndim_ == 3 && dtype_ == INFINI_DTYPE_F32"`

### `_generate_suffix(values)`
- **位置**: `build.py` (第 104-105 行)
- **功能**: 将参数值列表转换为下划线分隔的字符串后缀
- **用途**: 生成唯一的 kernel 标识符

### `_generate_param_value_combinations(param_grid)`
- **位置**: `build.py` (第 108-112 行)
- **功能**: 计算参数网格的笛卡尔积
- **算法**:
  - 提取所有参数名作为键列表
  - 使用 `itertools.product(*param_grid.values())` 生成所有可能的值组合
  - 将每个组合转换为字典（键值对映射）
- **时间复杂度**: O(Π n_i)，其中 n_i 是第 i 个参数的取值数量（笛卡尔积大小）

## 3. API 接口

```python
def build(
    premake: Callable,                        # 算子预构建函数
    constexpr_param_grid: Dict[str, Iterable], # 编译期常量参数网格
    caller: str,                              # 目标平台标识
    op_name: str,                             # 算子名称
    output_dir: pathlib.Path                  # 输出目录路径
) -> None
# 为算子的所有参数组合生成 CUDA kernel 和 C 语言接口
```

```c
// 生成的 C 接口示例（以 ReLU 为例）
NineToothedResult launch_relu(
    NineToothedStream stream,
    NineToothedTensor input,
    NineToothedTensor output,
    int ndim_,
    int dtype_
);

// 返回值：NineToothedResult 实际映射到 infiniStatus_t 枚举类型
//   - INFINI_STATUS_SUCCESS (0): 成功找到匹配的 kernel 并执行
//   - INFINI_STATUS_NOT_IMPLEMENTED (2): 没有匹配的参数组合
```

## 4. 使用示例

```python
# 示例：为 ReLU 算子生成多版本 kernel
import ninetoothed
from ntops.kernels import relu
import infiniop.ninetoothed.build

def build():
    MAX_NDIM = 5

    # 定义参数搜索空间
    ndim_values = range(1, MAX_NDIM + 1)          # 支持 1-5 维张量
    dtype_values = (
        ninetoothed.float16,                      # 半精度浮点
        ninetoothed.bfloat16,                     # 脑浮点
        ninetoothed.float32,                      # 单精度浮点
        ninetoothed.float64,                      # 双精度浮点
    )

    constexpr_param_grid = {
        "ndim": ndim_values,                       # 张量维度
        "dtype": dtype_values,                     # 数据类型
        "block_size": (1024,),                     # CUDA 块大小
    }

    # 调用构建函数
    infiniop.ninetoothed.build.build(
        relu.premake,                             # ReLU 算子的 premake 函数
        constexpr_param_grid,
        caller="cuda",                            # 生成 CUDA 代码
        op_name="relu",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH
    )

# 生成的代码结构：
# 1. relu_1_1024_NINETOOTHED_FLOAT16.h
# 2. relu_2_1024_NINETOOTHED_FLOAT32.h
# 3. ... (共 5 × 4 = 20 个 kernel 变体)
# 4. relu.c - 包含 launch_relu() 调度函数
# 5. relu.h - 声明 C 接口
```

生成的 C 代码示例：
```c
// relu.c 中的自动生成代码片段
NineToothedResult launch_relu(
    NineToothedStream stream,
    NineToothedTensor input,
    NineToothedTensor output,
    int ndim_,
    int dtype_
) {
    if (ndim_ == 1 && dtype_ == NINETOOTHED_FLOAT16 && block_size_ == 1024)
        return launch_relu_1_1024_NINETOOTHED_FLOAT16(stream, input, output);
    if (ndim_ == 2 && dtype_ == NINETOOTHED_FLOAT32 && block_size_ == 1024)
        return launch_relu_2_1024_NINETOOTHED_FLOAT32(stream, input, output);
    // ... 其他 18 个条件分支
    return INFINI_STATUS_NOT_IMPLEMENTED;
}
```

## 5. 实现细节

### 代码生成流程
1. **参数空间遍历**: 通过 `itertools.product` 实现完整的参数网格枚举，确保覆盖所有支持的参数组合
2. **符号转换**: 将 Python 的 dtype 对象（如 `ninetoothed.float16`）转换为 C 宏定义（`NINETOOTHED_FLOAT16`）
3. **命名规范化**: 为每个 kernel 生成全局唯一的标识符，避免链接冲突
4. **接口封装**: 自动生成符合 InfiniOP C API 规范的函数声明和实现

### 内存管理
- **文件输出**: 使用 `pathlib.Path.write_text()` 原子性写入生成的源代码文件
- **构建目录**: 默认输出到 `{InfiniCore}/build/ninetoothed/`，由 `BUILD_DIRECTORY_PATH` 常量指定

### 类型系统
- **dtype 映射**: 支持以下数据类型转换
  - 浮点型: `float16`, `bfloat16`, `float32`, `float64`
  - 整型: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
- **C 类型封装**: 所有张量参数统一封装为 `NineToothedTensor` 结构体（包含 `data`, `shape`, `strides` 字段）

### 并发性
- **独立编译**: 每个参数组合生成独立的 kernel，可以并行编译（通过 `make -j` 加速）
- **无共享状态**: 构建过程无全局可变状态，支持多进程并行构建不同算子

### 性能优化
- **编译期特化**: 将运行时参数（如 `ndim`, `dtype`）提升为编译期常量，允许编译器进行激进优化
- **条件分支优化**: 生成的调度函数使用线性 if-else 链，CPU 分支预测器可以高效处理
- **内核函数签名**: 生成的 kernel 函数签名经过精心设计，所有张量参数通过指针传递，避免拷贝开销

### 错误处理
- **默认返回值**: 当没有匹配的参数组合时，返回 `INFINI_STATUS_NOT_IMPLEMENTED`，调用方可以据此实现 fallback 逻辑
- **类型安全**: 使用 `inspect.signature()` 提取函数参数类型，确保生成的 C 接口与 Python 函数签名一致

### 依赖关系
- **外部依赖**:
  - `ninetoothed`: 底层 CUDA kernel 代码生成器（本项目的外部依赖）
  - `ntops`: 算子内核库（提供 `premake` 函数）
- **生成产物依赖**:
  - `infinicore.h`: 提供 `infiniStatus_t` 枚举和 `INFINI_DTYPE_*` 宏定义
  - `ninetoothed.h`: 提供 `NineToothedTensor`, `NineToothedStream`, `NineToothedResult` 类型定义

### 设计模式
- **模板方法模式**: `build()` 定义算法骨架，`premake` 提供可变部分（算子逻辑）
- **策略模式**: 不同的 `caller` 参数（`cuda`, `cpu`）对应不同的代码生成策略
- **建造者模式**: 通过参数网格逐步构建复杂的 kernel 矩阵

### 构建产物
生成的文件遵循严格的命名约定：
- **源文件**: `{op_name}.c` - 包含所有 kernel 的 #include 和调度函数实现
- **头文件**: `{op_name}.h` - 包含 `extern "C"` 包装的函数声明
- **kernel 变体**: `{op_name}_{suffix1}_{suffix2}_...{suffixN}.h` - 每个参数组合对应的 kernel 头文件

该模块是 InfiniOP 自动化构建流程的核心组件，显著简化了 GPU 算子的开发流程，使得开发者只需定义算子的数学逻辑和参数空间，即可自动获得高度优化的 CUDA 实现和标准 C 接口。
