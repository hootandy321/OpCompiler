# testcases 目录代码分析报告

## 1. 目录概述

`testcases` 目录是 InfiniCore 测试框架中的**测试用例生成器集合**,负责为各种算子操作生成标准化的测试数据。该目录包含 16 个独立的测试用例生成模块,每个模块对应一个具体的算子操作,通过统一的测试框架(`InfiniopTestWriter` 和 `InfiniopTestCase`)生成 GGUF 格式的测试数据文件。

**核心职责**:
- 为不同算子生成自动化测试用例
- 提供参考实现用于验证计算结果
- 支持多种数据类型、张量形状和内存布局(步长)的测试
- 生成包含输入、输出和预期结果的标准化测试文件

---

## 2. 文件结构

该目录包含以下测试用例文件(按字母顺序):

| 文件名 | 算子类型 | 功能描述 |
|--------|---------|---------|
| `add.py` | 算术运算 | 张量加法测试 |
| `causal_softmax.py` | 激活函数 | 因果掩码 Softmax(用于注意力机制) |
| `clip.py` | 张量操作 | 值裁剪到指定范围 |
| `gemm.py` | 矩阵运算 | 通用矩阵乘法(GEMM) |
| `mul.py` | 算术运算 | 张量逐元素乘法 |
| `ones.py` | 张量创建 | 生成全1张量 |
| `random_sample.py` | 采样操作 | Top-p/Top-k 随机采样(用于文本生成) |
| `rearrange.py` | 内存操作 | 张量重排(行优先/列优先转换) |
| `rms_norm.py` | 归一化 | RMS Layer Normalization |
| `rope.py` | 位置编码 | 旋转位置编码(RoPE) |
| `sigmoid.py` | 激活函数 | Sigmoid 激活函数 |
| `sub.py` | 算术运算 | 张量减法 |
| `swiglu.py` | 激活函数 | SwiGLU 激活函数(用于 LLaMA 等) |
| `topkrouter.py` | MoE 路由 | DeepSeek-V3 Top-K 专家路由 |
| `topksoftmax.py` | MoE 路由 | Top-K Softmax 路由 |
| `zeros.py` | 张量创建 | 生成全0张量 |
| `__init__.py` | 模块初始化 | 空文件,标识为 Python 包 |

---

## 3. 核心设计模式

### 3.1 统一测试框架

所有测试用例都继承自 `InfiniopTestCase` 基类,遵循一致的模式:

```python
class XxxTestCase(InfiniopTestCase):
    def __init__(self, 输入张量, 形状, 步长, 输出张量, ...):
        super().__init__("算子名称")
        # 存储测试参数

    def write_test(self, test_writer: "InfiniopTestWriter"):
        # 1. 写入形状和步长信息
        # 2. 写入输入/输出张量数据
        # 3. 计算并写入参考答案(通常使用 float64 精度)
```

### 3.2 测试配置模式

每个模块都定义了标准化的测试配置:

```python
_TEST_CASES_ = [
    # (shape, stride_a, stride_b, stride_c, ...)
    # 测试不同的形状组合和内存布局
]
_TENSOR_DTYPES_ = [np.float16, np.float32]  # 支持的数据类型
```

**常见的测试场景**:
- **基础形状**: 2D、3D 张量(如 `(13, 4)`, `(2, 4, 2048)`)
- **LLM 相关形状**: 模拟实际模型工作负载(如 `(16, 5632)`, `(4, 48, 64)`)
- **步长模式**:
  - `None`: 连续内存(默认)
  - `(step, 1)`: 非连续内存布局
  - `(0, 1)`: 广播模式(某些维度步长为0)
- **数据类型**: `float16`, `float32`, 部分支持 `float64` 和整数类型

### 3.3 参考实现

每个测试用例都包含一个 NumPy/PyTorch 实现的参考函数:

```python
def xxx_ref_implementation(a, b, ...):
    # 使用 NumPy/PyTorch 实现的参考计算
    return result
```

参考实现用于:
1. 生成测试的"金标准"答案
2. 验证硬件后端的实现正确性
3. 通常使用 `float64` 精度以减少数值误差

---

## 4. 关键算子分类

### 4.1 算术运算 (4 个)

| 算子 | 文件 | 输入 | 输出 | 特殊处理 |
|------|------|------|------|---------|
| **Add** | `add.py` | `a, b` | `a + b` | 支持零步长广播 |
| **Sub** | `sub.py` | `a, b` | `a - b` | 支持零步长广播 |
| **Mul** | `mul.py` | `a, b` | `a * b` | 同时保存 float32 和 float64 结果 |
| **Clip** | `clip.py` | `x, min, max` | `clamp(x, min, max)` | 支持张量形式的 min/max |

**测试覆盖**:
- 形状: `(13, 4)`, `(13, 4, 4)`, `(16, 5632)`, `(4, 4, 5632)` (模拟 LLM 中间层)
- 步长: 连续内存、非连续内存、零步长广播
- 数据类型: `float16`, `float32`

### 4.2 激活函数 (4 个)

| 算子 | 文件 | 公式 | 应用场景 |
|------|------|------|---------|
| **Sigmoid** | `sigmoid.py` | `1 / (1 + e^(-x))` | 二分类、门控机制 |
| **SwiGLU** | `swiglu.py` | `a * sigmoid(b)` | LLaMA、PaLM 等现代 LLM |
| **Causal Softmax** | `causal_softmax.py` | `softmax(x * causal_mask)` | 因果注意力机制(如 GPT) |
| **RMS Norm** | `rms_norm.py` | `x / sqrt(mean(x²) + ε) * w` | LLaMA、T5 等 LayerNorm 替代 |

**特殊之处**:
- `Causal Softmax`: 实现因果掩码(只能看到当前位置之前的信息)
- `RMS Norm`: 测试形状涵盖 `(2, 256)` 到 `(500, 4096)` 的各种 hidden size
- `SwiGLU`: 部分测试用例使用 `(0, 1)` 步长测试广播

### 4.3 矩阵运算 (1 个)

| 算子 | 文件 | 功能 | 参数 |
|------|------|------|------|
| **GEMM** | `gemm.py` | 通用矩阵乘法 | `C = alpha * A @ B + beta * C` |

**测试场景**:
- 小矩阵: `(4, 5) @ (5, 6)`
- 大矩阵: `(1, 2048) @ (2048, 2048)`
- 批处理: `(2, 4, 2048) @ (2, 2048, 2048)`
- 特殊 alpha/beta: `(1.0/8, 1.0)`

**关键测试点**:
- 支持列优先存储(stride = `(1, rows)`)
- 批量矩阵乘法(3D 张量)
- 不同的 alpha 和 beta 系数

### 4.4 MoE (混合专家) 路由 (2 个)

这两个算子用于 MoE (Mixture of Experts) 架构:

#### 4.4.1 TopKRouter (`topkrouter.py`)

**功能**: DeepSeek-V3 风格的 Top-K 专家路由

**算法细节**:
1. 将输入 logits 经过 sigmoid 得到分数
2. 将专家分成 8 组,每组选 2 个最高分
3. 在选出的组内选择 Top-4 专家
4. 最终选择 Top-8 专家并归一化权重

**参数**:
- `x`: `(n_tokens, 256)` - 256 个专家的路由 logits
- `correction_bias`: `(256,)` - 专家分数修正偏置
- `routed_scaling_factor`: 缩放因子(如 2.5)
- `topk`: 最终选择的专家数(8)

#### 4.4.2 TopKSoftmax (`topksoftmax.py`)

**功能**: Mixtral 风格的 Top-K Softmax 路由

**算法细节**:
1. 对 router_logits 进行 softmax
2. 选择 Top-K 专家
3. 可选的归一化(`norm_topk_prob`)

**参数**:
- `x`: `(n_tokens, n_experts)`
- `topk`: 选择的专家数
- `norm`: 是否归一化权重

**实现依赖**: 使用 PyTorch 实现参考算法

### 4.5 位置编码 (1 个)

#### RoPE (`rope.py`)

**功能**: 旋转位置编码(Rotary Position Embedding)

**支持两种算法**:
1. **GPT-J**: 偶数/奇数维度交替处理
2. **GPT-NeoX**: 前/后半维度分别处理

**测试参数**:
- 形状: `(seq_len, n_heads, head_dim)`,如 `(1, 32, 128)`, `(10, 32, 64)`
- 特殊步长测试: `(4, 1, 32)` (步长大于尺寸,测试边界情况)

**辅助函数**:
```python
sin_cos_table(pos, dim, theta, dtype):
    # 生成 sin/cos 查找表
    # theta = 10000 (默认)
```

### 4.6 采样操作 (1 个)

#### RandomSample (`random_sample.py`)

**功能**: 从词汇表中采样一个 tokenID

**算法**:
1. 选择 Top-K 个最高分的 token
2. 在 Top-K 中应用 Top-p (nucleus sampling)
3. 根据累积概率分布进行随机采样

**参数**:
- `data`: `(voc_size,)` - 词汇表分数
- `topk`: 考虑的前 K 个候选(如 3, 5, 10, 50)
- `topp`: 累积概率阈值(如 0.8, 0.9)
- `temperature`: 温度参数(如 0.5, 1.0, 2.0)
- `random_val`: 随机数(用于测试确定性)

**测试词汇表大小**: 512, 4096, 16384, 32000

### 4.7 内存操作 (1 个)

#### Rearrange (`rearrange.py`)

**功能**: 改变张量的内存布局(步长)

**测试场景**:
- 行优先 ↔ 列优先转换
- 形状: `(100, 100)`, `(4, 6, 64)`, `(2000, 2000)`
- 高维张量: `(3, 4, 7, 53, 9)`, `(3, 4, 50, 50, 5, 7)`

**实现**: 使用 PyTorch 的 `set_` 方法修改步长

**辅助函数**:
```python
row_major_strides(shape)    # 行优先步长
column_major_strides(shape) # 列优先步长
```

### 4.8 张量创建 (2 个)

| 算子 | 文件 | 输出 | 支持的数据类型 |
|------|------|------|---------------|
| **Ones** | `ones.py` | 全1张量 | bool, int8-64, float16/32/64 |
| **Zeros** | `zeros.py` | 全0张量 | bool, int8-64, float16/32/64 |

**测试覆盖**:
- 形状: 2D、3D
- 数据类型: 包括布尔和整数类型(罕见)
- 步长: 连续和非连续内存

---

## 5. 公共工具函数

### 5.1 随机张量生成

```python
def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate  # 数值范围在 [-5e-4, 5e-4]
    return rate * np.random.rand(*shape).astype(dtype) - var
```

**特点**:
- 生成小范围随机值,避免数值溢出
- 用于大部分测试用例的输入数据

### 5.2 零步长张量处理

```python
def process_zero_stride_tensor(tensor, stride):
    if stride and 0 in stride:
        # 对零步长维度进行切片,避免数据重叠
        slices = tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)
        return tensor[slices]
    return tensor
```

**用途**: 处理广播场景(步长为0表示在该维度广播)

### 5.3 步长转换

```python
def gguf_strides(*strides):
    # 将 Python 步长转换为 GGUF 格式

def contiguous_gguf_strides(shape):
    # 计算连续内存的默认步长(行优先)
```

---

## 6. 测试文件生成流程

每个测试文件执行时:

1. **创建 TestWriter**:
   ```python
   test_writer = InfiniopTestWriter("算子名称.gguf")
   ```

2. **生成测试用例**:
   ```python
   for dtype in _TENSOR_DTYPES_:
       for shape, strides in _TEST_CASES_:
           # 创建输入张量
           # 创建输出张量(空)
           # 创建 TestCase 对象
           test_cases.append(test_case)
   ```

3. **写入数据**:
   ```python
   test_writer.add_tests(test_cases)
   test_writer.save()  # 保存为 GGUF 文件
   ```

**生成的 GGUF 文件包含**:
- 张量形状(shape)
- 内存步长(strides)
- 输入数据(如 `a`, `b`, `x`)
- 输出占位符(如 `c`, `y`)
- 参考答案(`ans`,通常为 float64)

---

## 7. 特殊测试场景

### 7.1 零步长广播测试

**涉及的算子**: `add`, `sub`, `sigmoid`, `swiglu`, `zeros`, `ones`

**示例**:
```python
# shape=(13, 4), stride=(0, 1)
# 表示在维度0上广播(所有行共享数据)
```

**验证点**:
- 正确处理广播语义
- 避免内存重叠导致的错误

### 7.2 大规模张量测试

**GEMM**:
- `(2048, 2048)` - 测试大矩阵乘法性能
- `(2, 4, 2048)` - 批量大矩阵

**RMS Norm**:
- `(500, 4096)` - 模拟大批量、大 hidden size

### 7.3 非连续内存测试

**所有算子都支持**:
```python
# shape=(13, 4), stride=(10, 1)
# 表示维度0的步长为10(大于尺寸4),内存不连续
```

**验证点**:
- 正确解析非连续内存
- 避免越界访问

### 7.4 多精度测试

**大部分算子**: `float16`, `float32`
**部分算子**: `float64`(用于参考答案)
**Ones/Zeros**: 额外支持 `bool`, `int8-64`

### 7.5 算法变体测试

**RoPE**: GPT-J vs GPT-NeoX 两种算法
**TopKRouter**: 不同的 `routed_scaling_factor` 和 `topk`
**RandomSample**: 不同的 `topp`, `topk`, `temperature` 组合

---

## 8. 与测试框架的集成

### 8.1 InfiniopTestCase 基类

所有测试用例继承的基类,提供:
- 统一的构造函数接口
- `write_test()` 方法模板
- GGUF 键名生成(`gguf_key()`)

### 8.2 InfiniopTestWriter

负责写入测试数据的工具类:
- 管理 GGUF 文件格式
- 提供类型安全的数据写入方法:
  - `add_tensor()`: 写入张量数据
  - `add_array()`: 写入数组(如形状、步长)
  - `add_float32()`, `add_int32()`, `add_bool()`: 写入标量参数

### 8.3 数据类型映射

```python
def np_dtype_to_ggml(dtype):
    # 将 NumPy 数据类型映射到 GGML/ GGUF 类型
    # 例如: np.float16 -> GGMLQuantizationType.F16
```

---

## 9. 测试覆盖的关键场景

### 9.1 LLM 推理核心算子

- **GEMM**: 线性层、注意力 QKV 计算
- **RMS Norm**: 层归一化
- **RoPE**: 位置编码
- **SwiGLU**: 前馈网络激活函数
- **Causal Softmax**: 因果注意力

### 9.2 文本生成算子

- **RandomSample**: Top-p/Top-k 采样
- **TopKRouter/TopKSoftmax**: MoE 专家路由

### 9.3 基础算子

- **Add/Sub/Mul**: 逐元素运算
- **Clip**: 激活值裁剪
- **Sigmoid**: 门控机制
- **Ones/Zeros**: 张量初始化
- **Rearrange**: 内存布局转换

### 9.4 边界情况

- 零步长(广播)
- 大步长(非连续内存)
- 小尺寸(如 `(1, 1)`)
- 大尺寸(如 `(2048, 2048)`)
- 质数维度(如 `(7, 13)`)

---

## 10. 代码质量特点

### 10.1 一致性

- 所有文件遵循相同的命名约定
- 统一的测试配置模式
- 一致的错误处理

### 10.2 可维护性

- 清晰的函数命名
- 适当的注释(特别是复杂算法如 RoPE)
- 分离参考实现和测试用例生成

### 10.3 可扩展性

- 添加新算子只需创建新文件
- 测试配置易于调整
- 支持新的数据类型和形状

### 10.4 验证完整性

- 每个算子都有参考实现
- 参考答案使用高精度(float64)
- 覆盖多种输入场景

---

## 11. 与父目录的关系

该目录是 `test_generate` 的子目录,专门负责**算子级别的测试用例生成**。父目录可能包含:
- 测试运行器
- 结果验证工具
- 性能基准测试

---

## 12. 总结

`testcases` 目录是一个**设计精良的测试用例生成系统**,具有以下特点:

1. **全面性**: 覆盖 LLM 推理的核心算子
2. **标准化**: 统一的测试框架和文件格式
3. **灵活性**: 支持多种数据类型、形状和内存布局
4. **可验证性**: 每个算子都有参考实现
5. **实用性**: 测试场景贴近实际工作负载

该目录是 InfiniCore 测试基础设施的**核心组件**,为硬件后端的正确性验证提供了可靠的数据基础。
