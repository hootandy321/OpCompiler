# ninetoothed/tests 核心实现文档

ninetoothed 测试套件提供全面的内核验证框架，覆盖从基础算术运算到复杂深度学习算子（如 Attention、MatMul、Conv2D）的完整测试体系。该测试套件采用 pytest 参数化测试模式，支持多设备（CUDA/MLU）和多种数据类型，确保 ninetoothed JIT 编译器生成的内核在功能上与 PyTorch 原生实现完全一致。

## 1. 模块结构

- **`conftest.py`**: pytest 配置文件，实现确定性的随机种子管理机制
- **`utils.py`**: 设备检测工具，提供 CUDA 和 MLU 设备可用性查询
- **`test_add.py`**: 基础加法运算测试，验证简单的逐元素加法内核
- **`test_addmm.py`**: 矩阵乘法加法融合算子测试（addmm），实现 `output = beta * input + alpha * (mat1 @ mat2)` 语义
- **`test_aot.py`**: Ahead-Of-Time 编译测试，验证内核预编译、动态库生成和加载流程
- **`test_attention.py`**: Flash Attention 算法测试，实现因果注意力机制的在线更新算法
- **`test_clone.py`**: 张量克隆操作测试，验证四种不同的内存复制实现方式
- **`test_conv2d.py`**: 二维卷积测试，通过 im2col 转换将卷积映射为矩阵乘法
- **`test_data_ptr.py`**: 数据指针操作测试，验证原子加法（atomic_add）和底层内存访问
- **`test_debugging.py`**: 调试工具测试，验证 arrangement 模拟和张量排列转换的正确性
- **`test_dropout.py`**: Dropout 正则化测试，实现训练时的随机神经元丢弃
- **`test_eval.py`**: 符号计算求值测试，验证张量表达式的符号求值和替换机制
- **`test_expand.py`**: 张量广播测试，验证零维张量的扩展操作
- **`test_generation.py`**: 内核生成策略测试，验证自动调优、block size 配置和代码生成
- **`test_getitem.py`**: 张量索引测试，验证多维切片和高级索引操作
- **`test_ipynb.py`**: Jupyter Notebook 集成测试，验证在 .ipynb 环境中的内核执行
- **`test_jagged.py`**: 锯齿张量（Jagged Tensor）测试，验证可变长度序列的处理
- **`test_matmul.py`**: 矩阵乘法核心测试，验证分块矩阵乘法算法（GEMM）
- **`test_max_pool2d.py`**: 二维最大池化测试，验证滑动窗口最大值提取
- **`test_naming.py`**: 命名规则测试，验证符号命名前缀和类型标记
- **`test_pad.py`**: 张量填充测试，验证常量填充模式
- **`test_pow.py`**: 幂运算测试，验证 libdevice 数学库调用
- **`test_softmax.py`**: Softmax 激活函数测试，验证数值稳定的指数归一化
- **`test_unsqueeze.py`**: 维度扩展测试，验证张量形状操作

## 2. 核心测试类

### `Arrangement-Application 模式`
所有算子测试遵循统一的"排列-应用"二元架构：

- **Arrangement 阶段**：负责内存布局转换和 tiling 策略，将输入张量分割为适合 GPU 并行处理的 block
- **Application 阶段**：实际计算逻辑，使用 ninetoothed.language (ntl) DSL 表式计算

### `参数化测试策略`
使用 `pytest.mark.parametrize` 实现多维度测试覆盖：

```python
@pytest.mark.parametrize("device", get_available_devices())  # CUDA/MLU
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))  # 数据类型
@pytest.mark.parametrize("size", (98432,))  # 问题规模
def test(size, dtype, device):
    # 测试逻辑
```

## 3. 关键测试实现

### `test_attention.py` - Flash Attention 测试
- **Location**: `test_attention.py`
- **Primary Function**: 验证 Flash Attention 在线算法的正确性，该算法通过迭代更新避免显式构建完整的注意力矩阵
- **Key Constants**:
  - `BLOCK_SIZE_M`: 可调，范围 [64, 128]，控制 query 序列的 block 大小
  - `BLOCK_SIZE_N`: 可调，范围 [32, 64]，控制 key/value 序列的 block 大小
- **Core Algorithm**:
  ```python
  # 在线 Flash Attention 算法
  acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
  l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
  m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

  for i in range(k.shape[0]):
      qk = ntl.dot(q_loaded, ntl.trans(k[i]))
      m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
      p = ntl.exp2(qk - m_ij[:, None])
      alpha = ntl.exp2(m_i - m_ij)
      acc = acc * alpha[:, None] + ntl.dot(p.to(v[i].dtype), v[i])
      m_i = m_ij
      l_i = l_i * alpha + l_ij

  acc /= l_i[:, None]
  ```
- **Complexity**: O(seq_len²) 时间复杂度，但通过 tiling 优化内存访问模式
- **Causal Masking**: 通过 `q.offsets(-2)[:, None] >= k[i].offsets(-2)[None, :]` 实现因果掩码

### `test_matmul.py` - 分块矩阵乘法测试
- **Location**: `test_matmul.py`
- **Primary Function**: 验证 GEMM（通用矩阵乘法）内核的正确性
- **Key Symbols**:
  - `BLOCK_SIZE_M`: meta 符号，控制输出矩阵行方向的 block 大小
  - `BLOCK_SIZE_N`: meta 符号，控制输出矩阵列方向的 block 大小
  - `BLOCK_SIZE_K`: meta 符号，控制累维度的 block 大小
- **Core Tiling Strategy**:
  ```python
  # 输出矩阵 tiling
  output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

  # 左矩阵 tiling (M x K)
  lhs_tiled = (
      lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
      .tile((1, -1))
      .expand((-1, output_tiled.shape[1]))
  )

  # 右矩阵 tiling (K x N)
  rhs_tiled = (
      rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
      .tile((-1, 1))
      .expand((output_tiled.shape[0], -1))
  )
  ```
- **Accumulation Pattern**:
  ```python
  accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
  for k in range(lhs.shape[0]):
      accumulator += ntl.dot(lhs[k], rhs[k])
  output = accumulator.to(ntl.float16)
  ```
- **Numerical Stability**: 使用 float32 累加器避免 float16 精度损失

### `test_conv2d.py` - 卷积转矩阵乘法测试
- **Location**: `test_conv2d.py`
- **Primary Function**: 验证通过 im2col 技术将卷积转换为矩阵乘法的正确性
- **Transformation Pipeline**:
  1. **Input Tiling**: 将输入张量按照滤波器形状进行分块
     ```python
     input_tiled = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
     ```
  2. **Squeeze**: 移除通道维度
     ```python
     input_squeezed = input_tiled.squeeze(1)
     ```
  3. **Ravel**: 将空间维度展平
     ```python
     input_raveled = input_squeezed.ravel()
     ```
  4. **Flatten**: 将批次和通道维度合并
     ```python
     input_flattened = input_raveled.flatten(end_dim=3).flatten(start_dim=1)
     ```
  5. **Filter Permutation**: 将滤波器转换为转置形式
     ```python
     filter_permuted = filter.flatten(start_dim=1).permute((1, 0))
     ```
- **Complexity**: 对于 (N, C, H, W) 输入和 (K, C, R, S) 滤波器，输出为 (N, K, P, Q)，计算复杂度为 O(N·K·C·P·Q·R·S)

### `test_addmm.py` - 矩阵乘法加法融合测试
- **Location**: `test_addmm.py`
- **Primary Function**: 验证 `output = beta * input + alpha * (mat1 @ mat2)` 融合算子
- **Numerical Precision**: 支持 float16 和 float8_e5m2（FP8）数据类型
- **Tolerance Configuration**:
  - float16: `atol=0.075`（绝对误差容忍度）
  - float8_e5m2: `atol=0.125`（更大的容忍度以适应低精度）
- **Reuse Strategy**: 复用 `test_matmul.py` 的 arrangement 和 application 逻辑

### `test_aot.py` - Ahead-Of-Time 编译测试
- **Location**: `test_aot.py`
- **Primary Function**: 验证内核的预编译、动态库生成和 ctypes 加载流程
- **Compilation Pipeline**:
  1. **Code Generation**: ninetoothed.make() 生成 C 内核代码
  2. **Compilation**: 调用 nvcc 编译为 .so 动态库
     ```python
     command = [
         "nvcc", "-shared", "-Xcompiler", "-fPIC",
         "-lcuda", "-o", f"{kernel_name}.so"
     ] + list(glob(f"{kernel_name}*.c"))
     subprocess.run(command, check=True)
     ```
  3. **Loading**: 使用 ctypes.CDLL 加载动态库
     ```python
     library = ctypes.CDLL(f"{kernel_name}.so")
     launch_func = getattr(library, f"launch_{kernel_name}")
     ```
  4. **Execution**: 通过 CUDA Stream 调用内核
     ```python
     stream = torch.cuda.Stream()
     launch_func(ctypes.c_void_p(stream.cuda_stream), *arguments)
     ```
- **Argument Marshaling**: 使用 `_ArgumentTensor` 结构体封装张量数据指针、形状和步长信息
  ```python
  class _ArgumentTensor(ctypes.Structure):
      _fields_ = [
          ("data", ctypes.c_void_p),
          ("shape", ctypes.POINTER(ctypes.c_uint64)),
          ("strides", ctypes.POINTER(ctypes.c_int64)),
      ]
  ```

### `test_jagged.py` - 锯齿张量测试
- **Location**: `test_jagged.py`
- **Primary Function**: 验证对可变长度序列（Jagged Tensor）的处理能力
- **Key Classes**:
  - `ToPaddedTensor`: 将锯齿张量转换为填充张量
  - `Copy`: 在目标锯齿张量上复制源张量数据
- **Jagged Dim Support**: 指定哪个维度是可变长度的（`jagged_dim` 参数）
- **Padding Strategy**: 使用填充值（如 -1）对齐不同长度的序列
- **Block Size Configuration**: 通过 `constexpr` 符号控制 block size
  ```python
  BLOCK_SIZE = Symbol("block_size", constexpr=True)
  ```

### `test_debugging.py` - 调试工具测试
- **Location**: `test_debugging.py`
- **Primary Function**: 验证 `ninetoothed.debugging.simulate_arrangement` 功能
- **Verification Strategy**:
  1. **Source Tensors**: 验证排列前的源张量形状
  2. **Target Tensors**: 验证排列后的目标张量形状
  3. **Reference Comparison**: 与手工构造的参考张量逐元素比较
- **Use Case**: 在实际编译前验证 arrangement 逻辑的正确性，避免运行时调试

## 4. 测试基础设施

### `conftest.py` - 确定性随机种子管理
- **Function**: `pytest_collectstart(collector)`
  - **Trigger**: 每个 pytest 模块收集时
  - **Action**: 根据模块名设置确定性随机种子
  - **Algorithm**: SHA256 哈希模块名，取模 2³² 作为种子
  ```python
  def _hash(string):
      return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2**32
  ```

- **Fixture**: `set_seed_per_module(request)`
  - **Scope**: module 级别
  - **Action**: 为每个测试模块设置唯一随机种子
  - **Benefit**: 同一模块内所有测试用例使用相同种子，保证可重现性

- **Fixture**: `set_seed_per_test(request)`
  - **Scope**: function 级别
  - **Autouse**: 自动应用于所有测试
  - **Action**: 为每个测试用例设置基于"模块路径::测试名称"的唯一种子
  - **Benefit**: 每个测试用例独立可重现，失败时可精确复现

### `utils.py` - 设备检测工具
- **Function**: `get_available_devices()`
  - **Return**: 元组，包含可用的加速设备标识
  - **Detection Logic**:
    ```python
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")
    return tuple(devices)
    ```
  - **MLU Support**: 条件导入 `torch_mlu`，支持寒武纪 MLU 设备
  - **Error Handling**: 使用 `contextlib.suppress` 优雅处理导入失败

## 5. 测试用例模式

### 基础算术运算测试模式
```python
def test(size, dtype, device):
    # 1. 生成随机输入
    input = torch.rand(size, dtype=dtype, device=device)
    other = torch.rand(size, dtype=dtype, device=device)

    # 2. 调用 ninetoothed 内核
    output = add(input, other)

    # 3. 计算 PyTorch 参考输出
    expected = input + other

    # 4. 验证数值一致性
    assert torch.allclose(output, expected)
```

### 复杂算子测试模式
```python
@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype, rtol, atol", ((torch.float32, 0.025, 0.025),))
@pytest.mark.parametrize("emb_dim", (64,))
@pytest.mark.parametrize("seq_len", (1024, 1))
@pytest.mark.parametrize("num_heads", (4,))
@pytest.mark.parametrize("batch_size", (2,))
def test(batch_size, num_heads, seq_len, emb_dim, dtype, device, is_causal, rtol, atol):
    # 多参数组合测试
    q, k, v = (
        torch.randn(batch_size, num_heads, seq_len, emb_dim, dtype=dtype, device=device)
        for _ in range(3)
    )

    output = attention(q, k, v, is_causal=is_causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=1)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
```

## 6. 使用示例

### 运行单个测试文件
```bash
# 运行 CUDA 设备上的 matmul 测试
pytest tests/test_matmul.py -k test -v

# 运行特定参数组合
pytest tests/test_attention.py::test[True-cuda-float32-64-1024-4-2-0.025-0.025]
```

### 运行多设备测试
```bash
# 运行所有可用设备
pytest tests/test_add.py -v

# 仅运行 CUDA 测试
pytest tests/ -k cuda -v

# 仅运行 MLU 测试
pytest tests/ -k mlu -v
```

### 调试失败的测试
```bash
# 显示详细输出
pytest tests/test_conv2d.py -vv -s

# 进入 pdb 调试器
pytest tests/test_matmul.py --pdb

# 仅运行上次失败的测试
pytest tests/ --lf
```

### 生成覆盖率报告
```bash
# 生成 HTML 覆盖率报告
pytest tests/ --cov=ninetoothed --cov-report=html

# 生成终端覆盖率报告
pytest tests/ --cov=ninetoothed --cov-report=term-missing
```

## 7. 实现细节

### 随机性控制
- **确定性策略**: 通过 SHA256 哈希实现从测试路径到种子的确定性映射
- **多层级隔离**: 模块级和测试级两层的种子隔离，避免干扰
- **全局状态同步**: 同时设置 `random.seed()` 和 `torch.manual_seed()`，确保 Python 随机数和 PyTorch 随机数同步

### 设备抽象
- **多后端支持**: 统一抽象 CUDA 和 MLU 设备
- **参数化设备**: 使用 `@pytest.mark.parametrize("device", get_available_devices())` 自动适配可用设备
- **条件跳过**: 使用 `pytest.skip()` 优雅处理不支持的后端

### 数值精度验证
- **相对误差容忍**: `rtol` 参数控制相对误差（如浮点运算精度损失）
- **绝对误差容忍**: `atol` 参数控制绝对误差（如接近零值的误差）
- **数据类型特定**: 不同数据类型使用不同的容忍度
  - float32: `atol=1e-8`
  - float16: `atol=0.001 ~ 0.075`
  - float8_e5m2: `atol=0.125`

### 内核缓存策略
- **动态编译缓存**: `test_pad.py` 使用 `_kernel_cache` 字典缓存已编译的内核
  ```python
  kernel_key = str(kernel_config)
  if kernel_key not in _kernel_cache:
      _kernel_cache[kernel_key] = make(arrangement, application, tensors)
  ```
- **配置键生成**: 基于张量维度和切片配置生成唯一键
- **性能优化**: 避免相同配置的重复编译开销

### AOT 编译流程
- **代码生成**: ninetoothed.make() 生成 C 代码文件（.c）
- **动态链接**: 使用 nvcc 编译为共享库（.so）
- **符号导出**: 内核入口点命名为 `launch_<kernel_name>`
- **运行时加载**: ctypes.CDLL 动态加载共享库
- **流管理**: 通过 torch.cuda.Stream 显式管理 CUDA 流

### Flash Attention 算法优化
- **内存复杂度优化**: 从 O(seq_len²) 降低到 O(seq_len * block_size)
- **数值稳定性**: 使用在线 max/exp 更新避免上溢/下溢
  ```python
  m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
  p = ntl.exp2(qk - m_ij[:, None])  # 减去最大值避免溢出
  ```
- **分块累加**: 通过迭代更新累加器减少内存占用
- **因果掩码优化**: 在计算过程中直接应用掩码，避免额外的掩码张量

### 锯齿张量处理
- **变长序列**: 支持批次内不同长度的序列
- **动态填充**: 自动计算最大序列长度并填充
- **Jagged Dim**: 指定哪个维度是可变长度的
- **Expand 操作**: 将标量或小张量扩展为锯齿张量形状

### 测试覆盖范围
- **基础算子**: add, pow, clone, expand, getitem, unsqueeze, pad
- **线性代数**: matmul, addmm
- **深度学习算子**: attention, conv2d, max_pool2d, softmax, dropout
- **高级特性**: jagged tensor, data_ptr 操作, AOT 编译, Jupyter 集成
- **工具链**: 符号求值、调试工具、命名规则、自动调优

### 并行测试执行
- **pytest-xdist**: 支持 `-n` 参数并行执行测试
  ```bash
  pytest tests/ -n 8  # 使用 8 个并行进程
  ```
- **设备隔离**: 每个进程独立管理设备资源
- **确定性保证**: 即使并行执行，每个测试的随机种子仍然独立且确定性

### 性能基准测试
- **问题规模参数化**: 使用 `@pytest.mark.parametrize` 测试不同规模
  ```python
  @pytest.mark.parametrize("size", (98432, 196864, 393728))
  ```
- **自动调优验证**: `test_generation.py` 验证自动调优是否生成正确代码
  ```python
  num_warps=(4, 8)  # 测试多个 warp 配置
  num_stages=(1, 3) # 测试多个 pipeline stage 配置
  ```
- **编译验证**: 检查生成的内核源码包含预期的优化标志

### 错误处理机制
- **优雅降级**: 不支持的设备自动跳过测试
- **详细错误信息**: 使用 `-vv` 显示完整的断言失败信息
- **资源清理**: 测试后自动清理临时文件（如 .ipynb 测试）
- **超时控制**: 使用 `pytest-timeout` 插件防止测试挂起

### 依赖管理
- **torch**: 核心依赖，提供张量计算和设备抽象
- **pytest**: 测试框架
- **numpy**: 用于数值比较和参考实现
- **torch_mlu**: 可选依赖，支持寒武纪 MLU 设备
- **nvcc**: AOT 编译测试需要 NVIDIA CUDA 编译器
- **jupyter/jupytext**: Jupyter Notebook 集成测试需要

### 测试数据生成
- **随机初始化**: 使用 `torch.randn` 和 `torch.rand` 生成测试数据
- **确定性种子**: 通过 conftest.py 保证每次运行生成的数据相同
- **边界情况**: 测试零维张量、空张量、单元素张量等边界情况
- **特殊值**: 测试 inf, -inf, nan, pi 等特殊数学值
