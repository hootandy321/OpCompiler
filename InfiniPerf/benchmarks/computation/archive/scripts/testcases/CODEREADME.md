# GEMM 测试用例生成器核心实现文档

本模块是 InfiniPerf 性能测试框架中用于生成通用矩阵乘法（GEMM, General Matrix Multiply）测试用例的代码生成器。该模块通过 NumPy 计算参考结果，并将测试数据序列化为 GGUF 格式文件，用于 Infiniop 算子库的性能验证和正确性测试。

## 1. 模块结构

- **`gemm.py`**: GEMM 测试用例生成器的核心实现，包含参考算法、测试用例类定义和测试数据生成逻辑

## 2. 核心类与函数

### `gemm()` 函数
- **位置**: `gemm.py` 第 9-18 行
- **主要功能**: 实现标准的 GEMM 算法参考实现，计算公式为 `C = α * A × B + β * C`
- **签名**:
  ```python
  def gemm(
      a: np.ndarray,
      b: np.ndarray,
      alpha: float = 1.0,
      c: np.ndarray = None,
      beta: float = 0.0,
  ) -> np.ndarray
  ```
- **算法细节**:
  - 当 `c` 为 `None` 时，执行简化计算 `α * np.matmul(a, b)`
  - 否则执行完整计算 `α * np.matmul(a, b) + β * c`
  - 使用 NumPy 的 `matmul()` 函数进行矩阵乘法，确保 O(n³) 的标准矩阵乘法复杂度
- **精度保证**: 测试用例生成时会将输入转换为 `float64` 进行计算，确保参考结果的数值精度

### `random_tensor()` 函数
- **位置**: `gemm.py` 第 21-24 行
- **主要功能**: 生成具有特定数值范围的小值随机张量，用于测试数值稳定性
- **签名**:
  ```python
  def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray
  ```
- **数值范围策略**:
  - 使用 `rate = 1e-3` 作为缩放因子
  - 数值范围在 `[-5e-4, 5e-4]` 之间（`rate * [0, 1) - 0.5 * rate`）
  - 小数值设计用于测试浮点运算的精度表现和溢出行为

### `GemmTestCase` 类
- **位置**: `gemm.py` 第 27-77 行
- **继承关系**: `InfiniopTestCase`
- **主要功能**: 封装单个 GEMM 测试用例的配置、数据写入和参考结果生成逻辑

#### 关键成员变量
- **`a: np.ndarray`**: 左矩阵输入（shape: `[M, K]`）
- **`stride_a: List[int] | None`**: 矩阵 A 的步长信息（用于广播或子矩阵测试）
- **`b: np.ndarray`**: 右矩阵输入（shape: `[K, N]`）
- **`stride_b: List[int] | None`**: 矩阵 B 的步长信息
- **`c: np.ndarray`**: 累加矩阵输入（shape: `[M, N]`）
- **`stride_c: List[int] | None`**: 矩阵 C 的步长信息
- **`alpha: float`**: 矩阵乘积的缩放因子
- **`beta: float`**: 矩阵 C 的缩放因子

#### 核心方法

**`__init__()`** (第 28-47 行)
- 初始化测试用例的所有参数
- 调用父类构造函数，传入操作名称 `"gemm"`

**`write_test()`** (第 49-77 行)
- **功能**: 将测试用例数据序列化到 GGUF 文件
- **写入顺序**:
  1. 调用 `super().write_test()` 写入操作名称（`op_name = "gemm"`）
  2. 条件写入步长信息（仅当步长非 `None` 时）:
     - `a.strides`: 矩阵 A 的字节步长或维度步长
     - `b.strides`: 矩阵 B 的步长
     - `c.strides`: 矩阵 C 的步长
  3. 写入标量参数:
     - `alpha`: Float32 格式
     - `beta`: Float32 格式
  4. 写入输入张量:
     - `a`: 使用 `np_dtype_to_ggml()` 转换数据类型
     - `b`: 同上
     - `c`: 同上
  5. 计算并写入参考答案:
     - 将输入转换为 `float64` 精度
     - 调用 `gemm()` 函数计算结果
     - 以 `F64` 格式存储（`gguf.GGMLQuantizationType.F64`）

## 3. API 接口

```python
# GEMM 参考实现
def gemm(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    c: np.ndarray = None,
    beta: float = 0.0,
) -> np.ndarray
# 计算 C = α*A×B + β*C，返回结果矩阵

# 随机测试数据生成
def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray
# 生成小值范围 [-5e-4, 5e-4] 的随机张量

# 测试用例类
class GemmTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
        alpha: float,
        beta: float,
    )
    # 初始化 GEMM 测试用例

    def write_test(self, test_writer: "InfiniopTestWriter")
    # 将测试数据写入 GGUF 文件
```

## 4. 使用示例

```python
# 示例：生成 GEMM 测试用例集并保存为 GGUF 文件
from testcases.gemm import GemmTestCase, random_tensor
from infiniop_test import InfiniopTestWriter
import numpy as np

# 创建测试文件写入器
test_writer = InfiniopTestWriter("gemm.gguf")

# 定义测试用例列表
test_cases = [
    # 大规模方阵测试（Float32）
    GemmTestCase(
        a=random_tensor((8192, 8192), np.float32),
        stride_a=None,
        b=random_tensor((8192, 8192), np.float32),
        stride_b=None,
        c=random_tensor((8192, 8192), np.float32),
        stride_c=None,
        alpha=1.0,
        beta=0.0,
    ),

    # 大规模方阵测试（Float16）
    GemmTestCase(
        a=random_tensor((8192, 8192), np.float16),
        stride_a=None,
        b=random_tensor((8192, 8192), np.float16),
        stride_b=None,
        c=random_tensor((8192, 8192), np.float16),
        stride_c=None,
        alpha=1.0,
        beta=0.0,
    ),

    # 非方阵测试（M=512, K=5120, N=5120）
    GemmTestCase(
        a=random_tensor((512, 5120), np.float32),
        stride_a=None,
        b=random_tensor((5120, 5120), np.float32),
        stride_b=None,
        c=random_tensor((512, 5120), np.float32),
        stride_c=None,
        alpha=1.0,
        beta=0.0,
    ),
]

# 批量添加测试用例
test_writer.add_tests(test_cases)

# 保存为 GGUF 文件（包含所有元数据和张量数据）
test_writer.save()
```

**执行脚本生成测试文件**:
```bash
# 直接运行 gemm.py 生成测试数据
python gemm.py
# 输出：gemm.gguf 文件，包含 6 个测试用例的完整数据
```

## 5. 实现细节

### 数据类型映射
- 使用 `np_dtype_to_ggml()` 函数进行 NumPy 类型到 GGML 类型的转换：
  - `np.float16` → `GGMLQuantizationType.F16`
  - `np.float32` → `GGMLQuantizationType.F32`
  - `np.float64` → `GGMLQuantizationType.F64`
  - `bfloat16` → `GGMLQuantizationType.BF16`
  - 整数类型支持 `I8`, `I16`, `I32`, `I64`

### 测试用例设计策略
当前 `gemm.py` 的 `__main__` 块定义了 6 个测试用例，覆盖以下场景：

1. **大规模方阵** (8192×8192):
   - Float32 版本：测试标准单精度性能
   - Float16 版本：测试半精度性能和数值稳定性

2. **中等非方阵** (512×5120 × 5120×5120 → 512×5120):
   - Float32 版本
   - Float16 版本
   - 模拟 Transformer 注意力矩阵尺寸

3. **长矩形矩阵** (512×5120 × 5120×13824 → 512×13824):
   - Float32 版本
   - Float16 版本
   - 测试非对称维度和内存访问模式

### GGUF 文件结构
生成的 `gemm.gguf` 文件采用键值存储格式：
- **元数据字段**:
  - `test.{i}.op_name`: 操作名称（"gemm"）
  - `test.{i}.a.strides`: 矩阵 A 的步长（可选）
  - `test.{i}.b.strides`: 矩阵 B 的步长（可选）
  - `test.{i}.c.strides`: 矩阵 C 的步长（可选）
  - `test.{i}.alpha`: α 参数（Float32）
  - `test.{i}.beta`: β 参数（Float32）

- **张量数据**:
  - `test.{i}.a`: 输入矩阵 A
  - `test.{i}.b`: 输入矩阵 B
  - `test.{i}.c`: 输入矩阵 C（累加器）
  - `test.{i}.ans`: 参考答案（Float64 精度）

- **全局字段**:
  - `test_count`: 总测试用例数量（Uint64）

### 数值精度保证
- **参考计算**: 所有测试用例的参考答案均使用 `float64` 精度计算，避免浮点误差累积
- **输入精度**: 测试输入保持原始精度（`float16` 或 `float32`），测试低精度输入的数值稳定性
- **小数值范围**: 使用 `[-5e-4, 5e-4]` 的随机值，测试算法在极端数值下的表现

### 步长（Stride）支持
- `stride_a`, `stride_b`, `stride_c` 参数支持非连续内存布局测试
- 当 `stride` 为 `None` 时，假设张量是行主序连续存储（C 风格）
- 步长参数可用于测试：
  - 矩阵转置（通过调整步长实现零拷贝转置）
  - 矩阵切片（通过设置步长为 0 实现广播）
  - 子矩阵运算

### 依赖关系
- **外部依赖**:
  - `numpy`: 矩阵运算和随机数生成
  - `gguf`: GGUF 文件格式序列化
  - `ml_dtypes.bfloat16`: BFloat16 数据类型支持

- **内部依赖**:
  - `..` (父目录模块): 导入 `InfiniopTestWriter`, `InfiniopTestCase`, `np_dtype_to_ggml`, `gguf_strides`
  - 基类 `InfiniopTestCase` 定义在 `/home/qy/src/Infini/InfiniCore/test/infiniop-test/test_generate/infiniop_test.py`

### 性能考虑
- **测试规模**: 使用 8192×8192 的大矩阵（约 256MB 单张矩阵），充分测试内存带宽和计算吞吐
- **数据生成**: `random_tensor()` 使用向量化 NumPy 操作，避免 Python 循环
- **文件写入**: GGUF 格式支持高效序列化，张量数据以二进制块存储

### 设计模式
- **模板方法模式**: `InfiniopTestCase` 定义测试用例接口，`GemmTestCase` 实现具体逻辑
- **构建器模式**: `InfiniopTestWriter` 负责构建 GGUF 文件，支持批量添加测试用例
- **策略模式**: 通过 `stride` 参数支持不同的内存布局策略

## 6. 测试场景覆盖

当前测试用例覆盖以下关键场景：

1. **数据精度**: Float16 和 Float32
2. **矩阵形状**:
   - 方阵 (8192×8192)
   - 高方阵 (512×5120 × 5120×5120)
   - 扁平矩形 (512×5120 × 5120×13824)
3. **参数组合**:
   - `α=1.0, β=0.0`: 纯矩阵乘法（无累加）
   - 未来可扩展 `β≠0` 测试累加场景

## 7. 扩展性设计

该模块设计支持以下扩展：

1. **添加新的测试用例**: 在 `__main__` 块的 `test_cases` 列表中添加 `GemmTestCase` 实例
2. **自定义数值范围**: 修改 `random_tensor()` 函数的 `rate` 参数
3. **支持其他精度**: 通过 `np_dtype_to_ggml()` 添加新的数据类型映射
4. **测试步长**: 为 `stride_a`, `stride_b`, `stride_c` 传入非 `None` 值
5. **累加测试**: 设置 `beta≠0` 测试 `C += α*A×B` 场景
