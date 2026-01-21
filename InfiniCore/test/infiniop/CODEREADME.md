# InfiniOP 测试框架核心实现文档

本测试框架为 InfiniOP 算子库提供全面的 Python 绑定、测试基础设施和 correctness 验证套件。它支持多种硬件后端（NVIDIA GPU、Ascend NPU、Cambricon MLU 等），提供统一的 CTypes FFI 接口、自动化测试流程、性能分析和调试工具。

## 1. 模块结构

- **`libinfiniop/`**: 核心测试基础设施模块
  - **`liboperators.py`**: 库加载与 CTypes 绑定，管理 libinfiniop.so/libinfinirt.so 动态链接
  - **`datatypes.py`**: 数据类型枚举定义（F16, BF16, F32, I32, I64 等 20+ 种类型）
  - **`devices.py`**: 硬件设备枚举（CPU, NVIDIA, ASCEND, CAMBRICON, METAX, MOORE, KUNLUN, HYGON, ILUVATAR, QY）
  - **`structs.py`**: C 结构体绑定（TensorDescriptor, Handle, OperatorDescriptor）
  - **`op_register.py`**: 算子函数签名注册表，为 35+ 算子定义 argtypes/restype
  - **`utils.py`**: 测试工具类（TestTensor, TestWorkspace）、设备同步、调试输出、性能分析

- **测试脚本（35+ 个算子）**:
  - **基础算子**: `add.py`, `sub.py`, `mul.py`, `div.py`（逐元素算术运算）
  - **矩阵运算**: `gemm.py`, `conv.py`（1D/2D/3D 卷积, GEMM）
  - **归一化**: `layer_norm.py`, `rms_norm.py`, `add_rms_norm.py`
  - **激活函数**: `relu.py`, `gelu.py`, `silu.py`, `sigmoid.py`, `tanh.py`, `swiglu.py`, `softplus.py`
  - **注意力机制**: `attention.py`, `paged_attention.py`, `paged_attention_prefill.py`, `paged_caching.py`, `paged_caching_prefill.py`
  - **位置编码**: `rope.py`（支持 GPT-J/GPT-NeoX 两种算法）
  - **采样与路由**: `random_sample.py`, `topksoftmax.py`, `topkrouter.py`
  - **特殊算子**: `dequantize_awq.py`, `scaled_mm_int8.py`, `clip.py`, `causal_softmax.py`, `logsoftmax.py`, `lp_norm.py`, `ones.py`, `zeros.py`, `rearrange.py`

## 2. 核心类

### `InfiniLib`
- **位置**: `libinfiniop/liboperators.py`
- **主要功能**: 封装 libinfiniop.so 和 libinfinirt.so 双库访问，提供统一属性查找接口
- **核心方法**:
  - `__getattr__(name)`: 自动在 libop 和 librt 间查找符号，优先 libop
- **生命周期**: 单例模式（`LIBINFINIOP`），在模块加载时通过 `open_lib()` 初始化

### `OpRegister`
- **位置**: `libinfiniop/op_register.py`
- **主要功能**: 装饰器注册模式，为 35+ 算子设置 CTypes 函数签名
- **核心方法**:
  - `operator(op)`: 类方法装饰器，将签名函数注册到 `registry` 列表
  - `register_lib(lib)`: 遍历注册表，为每个算子设置 argtypes/restype
- **设计模式**: 装饰器 + 注册表模式，实现自动化的 FFI 绑定生成

### `CTensor`
- **位置**: `libinfiniop/utils.py`
- **主要功能**: 封装 infiniopTensorDescriptor_t 的 C 对象，管理形状/步长/数据类型
- **核心成员**:
  - `descriptor`: ctypes 指针，指向 C 侧 TensorDescriptor
  - `dt`: InfiniDtype 枚举值
  - `ndim`: 张量维度数
  - `c_shape`: (c_size_t * ndim) 数组，存储形状
  - `c_strides`: (c_ssize_t * ndim) 数组，存储步长（支持广播、非连续内存）
- **核心方法**:
  - `destroy_desc()`: 释放 C 侧描述符，防止内存泄漏
- **生命周期**: RAII 模式，析构时自动调用 destroy_desc()

### `TestTensor(CTensor)`
- **位置**: `libinfiniop/utils.py`
- **主要功能**: 测试用张量封装，桥接 PyTorch 张量与 C 缓冲区，支持任意步长布局
- **核心成员**:
  - `_torch_tensor`: PyTorch 参考张量（用于生成 expected 值）
  - `_data_tensor`: 实际数据缓冲区（可能经过 rearrange_tensor 处理）
  - `device`: InfiniDeviceEnum 设备类型
  - `shape`, `strides`: 逻辑形状和步长
- **核心方法**:
  - `torch_tensor()`: 返回参考 PyTorch 张量（用于 correctness 验证）
  - `actual_tensor()`: 返回实际数据张量（可能非连续）
  - `data()`: 返回数据指针（void*），传递给 C 内核
  - `is_broadcast()`: 检测步长中是否包含 0（广播标记）
  - `from_torch()`: 工厂方法，从现有 PyTorch 张量创建 TestTensor
  - `from_binary()`: 工厂方法，从二进制文件加载（支持 NumPy dtype）
- **初始化模式**:
  - `random`: 随机初始化（整数类型用 randint，浮点用 rand）
  - `zeros`, `ones`: 全零/全一张量
  - `randint`: 整数随机初始化（用于量化算子）
  - `manual`: 从用户提供的 PyTorch 张量创建
  - `binary`: 从二进制文件加载
- **特殊处理**:
  - 广播支持：步长为 0 的维度会被压缩为 size=1
  - 非连续内存：通过 `rearrange_tensor()` 生成自定义步长布局
  - BF16 精度：调试时自动转 FP32 进行比对

### `TestWorkspace`
- **位置**: `libinfiniop/utils.py`
- **主要功能**: 算子工作空间内存管理
- **核心成员**:
  - `tensor`: TestTensor 对象（size > 0 时分配，size=0 时为 None）
  - `_size`: 工作空间字节数
- **核心方法**:
  - `data()`: 返回工作空间指针（可能为 None）
  - `size()`: 返回 ctypes.c_uint64(size)，用于 C 调用
- **优化**: 零大小工作空间不分配内存，避免无效分配

## 3. API 接口

### CTypes 库加载与初始化

```python
# libinfiniop/liboperators.py
def open_lib():
    """
    自动搜索 INFINI_ROOT 环境变量指定的目录加载 libinfiniop.so/libinfinirt.so
    支持平台检测：
      - Windows: bin/infiniop.dll, infinirt.dll
      - Linux: lib/libinfiniop.so, libinfinirt.so
    返回 InfiniLib 封装对象
    """
    INFINI_ROOT = os.getenv("INFINI_ROOT") or ~/.infini
    libop = ctypes.CDDLL(libop_path)
    librt = ctypes.CDDLL(librt_path)
    # 设置基础函数签名
    lib.infiniopCreateTensorDescriptor.argtypes = [...]
    lib.infiniopCreateTensorDescriptor.restype = c_int
    # ...

LIBINFINIOP = open_lib()  # 全局单例
```

### 算子签名注册

```python
# libinfiniop/op_register.py
@OpRegister.operator
def add_(lib):
    """
    为 Add 算子注册完整的 CTypes 函数签名
    在 open_lib() 后自动调用，设置 argtypes/restype
    """
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output
        infiniopTensorDescriptor_t,  # input_a
        infiniopTensorDescriptor_t,  # input_b
    ]

    lib.infiniopGetAddWorkspaceSize.restype = c_int32
    lib.infiniopGetAddWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # output
        c_void_p,  # input_a
        c_void_p,  # input_b
        c_void_p,  # stream (可选)
    ]

    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [infiniopOperatorDescriptor_t]
```

### 设备管理与同步

```python
# libinfiniop/utils.py
def create_handle():
    """创建 infiniopHandle_t，绑定到当前设备"""
    handle = infiniopHandle_t()
    check_error(LIBINFINIOP.infiniopCreateHandle(ctypes.byref(handle)))
    return handle

def destroy_handle(handle):
    """销毁句柄，释放设备资源"""
    check_error(LIBINFINIOP.infiniopDestroyHandle(handle))

def get_test_devices(args):
    """
    根据命令行参数选择测试设备
    支持：--cpu, --nvidia, --ascend, --cambricon, --metax, --moore, --kunlun, --hygon
    默认：CPU
    """
    devices_to_test = []
    if args.cpu: devices_to_test.append(InfiniDeviceEnum.CPU)
    if args.nvidia: devices_to_test.append(InfiniDeviceEnum.NVIDIA)
    # ... 其他设备
    return devices_to_test or [InfiniDeviceEnum.CPU]

def synchronize_device(torch_device):
    """设备同步，确保异步计算完成"""
    if torch_device == "cuda": torch.cuda.synchronize()
    elif torch_device == "npu": torch.npu.synchronize()
    elif torch_device == "mlu": torch.mlu.synchronize()
    elif torch_device == "musa": torch.musa.synchronize()

def get_sync_func(device):
    """返回设备同步函数（CPU/CAMBRICON 返回 None）"""
    if device in (InfiniDeviceEnum.CPU, InfiniDeviceEnum.CAMBRICON):
        return None
    return getattr(torch, torch_device_map[device]).synchronize
```

### 测试执行框架

```python
# libinfiniop/utils.py
def test_operator(device, test_func, test_cases, tensor_dtypes):
    """
    高级测试执行器，自动管理句柄和同步

    参数:
        device: InfiniDeviceEnum 设备类型
        test_func: 测试函数，签名为 test(handle, device, *test_case_params, dtype, sync)
        test_cases: 测试参数元组列表，每个元组会被解包传递给 test_func
        tensor_dtypes: 要测试的数据类型列表

    流程:
        1. 设置当前设备（infinirtSetDevice）
        2. 创建 infiniopHandle_t
        3. 过滤不支持的数据类型（如 Kunlun 不支持 BF16）
        4. 遍历所有 test_cases × tensor_dtypes 组合
        5. 调用 test_func，传入 handle、device、测试参数、dtype、sync
        6. 异常安全：finally 块中销毁 handle
    """
    LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
    handle = create_handle()
    tensor_dtypes = filter_tensor_dtypes_by_device(device, tensor_dtypes)
    try:
        for test_case in test_cases:
            for tensor_dtype in tensor_dtypes:
                test_func(handle, device, *test_case, tensor_dtype, get_sync_func(device))
    finally:
        destroy_handle(handle)
```

### 调试与验证

```python
# libinfiniop/utils.py
def debug(actual, desired, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    详细调试输出，对比 actual vs desired 张量

    功能:
        - BF16 自动转 FP32 精度比对
        - 计算差异 mask（基于 atol + rtol * abs(desired)）
        - 彩色终端输出不匹配元素（红/绿/蓝色）
        - 打印统计信息：dtype, atol, rtol, 不匹配数量/比例, min/max

    参数:
        actual: 实际输出张量
        desired: 期望参考张量（通常来自 PyTorch）
        atol: 绝对容忍度
        rtol: 相对容忍度
        equal_nan: NaN 是否视为相等
    """
    # BF16 特殊处理
    if actual.dtype == torch.bfloat16 or desired.dtype == torch.bfloat16:
        actual = actual.to(torch.float32)
        desired = desired.to(torch.float32)

    # 计算差异 mask
    diff_mask = nan_mismatch | (
        torch.abs(actual.to(torch.float64) - desired.to(torch.float64))
        > (atol + rtol * torch.abs(desired.to(torch.float64)))
    )

    # 彩色输出不匹配元素
    for idx in diff_indices:
        print(f" > Index: {idx} actual: \033[31m{val}\033[0m expect: \033[32m{exp}\033[0m delta: \033[33m{delta}\033[0m")

    # 统计信息
    print(f"  - Mismatched elements: {len(diff_indices)} / {actual.numel()} ({ratio}%)")

def debug_all(actual_vals, desired_vals, condition="or", atol=0, rtol=1e-2):
    """
    批量调试，支持多个值对的逻辑组合验证

    用途: RandomSample 等算子，允许多个正确答案（如 top-k 中任意一个都正确）
    """
    passed = False if condition == "or" else True
    for index, (actual, desired) in enumerate(zip(actual_vals, desired_vals)):
        if condition == "or":
            if not passed and len(print_discrepancy(...)) == 0:
                passed = True
        elif condition == "and":
            if passed and len(print_discrepancy(...)) != 0:
                passed = False
    assert passed
```

### 性能分析

```python
# libinfiniop/utils.py
def profile_operation(desc, func, torch_device, NUM_PRERUN, NUM_ITERATIONS):
    """
    统一的性能分析流程

    步骤:
        1. Warmup: 执行 NUM_PRERUN 次预热（避免冷启动影响）
        2. Timing: 执行 NUM_ITERATIONS 次计时，每次调用 synchronize_device()
        3. Output: 打印平均执行时间（毫秒）

    示例:
        profile_operation("PyTorch", lambda: torch_add(a, b), device, 10, 1000)
        profile_operation("    lib", lambda: lib_add(), device, 10, 1000)
        输出:
         PyTorch time: 0.123456 ms
             lib time: 0.098765 ms
    """
    # Warmup
    for _ in range(NUM_PRERUN):
        func()

    # Timed execution
    elapsed = timed_op(lambda: func(), NUM_ITERATIONS, torch_device)
    print(f" {desc} time: {elapsed * 1000:6f} ms")

def timed_op(func, num_iterations, device):
    """计时辅助函数，包含设备同步"""
    synchronize_device(device)
    start = time.time()
    for _ in range(num_iterations):
        func()
    synchronize_device(device)
    return (time.time() - start) / num_iterations
```

### 张量重排（非连续内存支持）

```python
# libinfiniop/utils.py
def rearrange_tensor(tensor, new_strides):
    """
    为 PyTorch 张量生成自定义步长布局

    算法:
        1. 根据 new_strides 计算新的物理大小（考虑广播维度）
        2. 创建全零缓冲区 tensor_zeros
        3. 使用 torch.meshgrid 生成原始张量的索引
        4. 根据新步长计算新的线性位置：new_pos = sum(idx[i] * new_strides[i])
        5. 使用 index_add_ 将原始数据散射到新位置
        6. 使用 torch.Tensor.set_() 设置自定义存储和步长

    应用场景:
        - 测试非连续内存访问模式
        - 验证内核对任意步长的正确性
        - 模拟广播张量（stride=0）

    限制:
        - 暂不支持负步长
        - 需要 dtype 支持 index_add_（特殊类型用 float64 中转）
    """
    shape = tensor.shape
    new_size = [(shape[i] - 1) * new_strides[i] + 1 for i in range(len(shape))]
    new_tensor = torch.zeros((sum(new_size),), dtype=tensor.dtype, device=tensor.device)

    # 生成索引网格
    indices = [torch.arange(s) for s in shape]
    mesh = torch.meshgrid(*indices, indexing='ij')
    linear_indices = [m.flatten() for m in mesh]

    # 计算新位置
    new_positions = sum(
        linear_indices[i] * new_strides[i] for i in range(len(shape))
    ).to(tensor.device)

    # 散射数据
    new_tensor.view(-1).index_add_(0, new_positions, tensor.contiguous().view(-1))

    # 设置自定义步长
    new_tensor.set_(new_tensor.untyped_storage(), 0, shape, tuple(new_strides))
    return new_tensor
```

## 4. 使用示例

### 编写新算子测试（以 Add 为例）

```python
# add.py
import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP, TestTensor, get_test_devices, check_error,
    test_operator, get_args, debug, get_tolerance, profile_operation,
    TestWorkspace, InfiniDtype, InfiniDtypeNames, InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# 1. 定义测试用例
_TEST_CASES_ = [
    ((13, 4), None, None, None),  # shape, a_stride, b_stride, c_stride
    ((13, 4), (10, 1), (10, 1), (10, 1)),  # 自定义步长
    ((13, 4), (0, 1), None, None),  # 广播测试
]

# 2. 定义数据类型和容忍度
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.I32, InfiniDtype.I64]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
}

# 3. 实现 PyTorch 参考函数
def add(c, a, b):
    torch.add(a, b, out=c)

# 4. 编写测试函数
def test(handle, device, shape, a_stride, b_stride, c_stride,
         inplace=Inplace.OUT_OF_PLACE, dtype=torch.float16, sync=None):
    print(f"Testing Add on {InfiniDeviceNames[device]} with shape:{shape} "
          f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
          f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}")

    # 创建测试张量
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride: return  # 步长不匹配，跳过
        c = a
    elif inplace == Inplace.INPLACE_B:
        if c_stride != b_stride: return
        c = b
    else:
        c = TestTensor(shape, c_stride, dtype, device, mode="ones")

    if c.is_broadcast():
        return  # 广播输出暂不支持

    # 计算 PyTorch 参考结果
    add(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())
    if sync is not None:
        sync()

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateAddDescriptor(
        handle, ctypes.byref(descriptor),
        c.descriptor, a.descriptor, b.descriptor
    ))

    # 销毁描述符（强制内核不依赖它）
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    # 获取并分配工作空间
    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetAddWorkspaceSize(
        descriptor, ctypes.byref(workspace_size)
    ))
    workspace = TestWorkspace(workspace_size.value, c.device)

    # 执行内核
    def lib_add():
        check_error(LIBINFINIOP.infiniopAdd(
            descriptor, workspace.data(), workspace.size(),
            c.data(), a.data(), b.data(), None
        ))
    lib_add()

    # 验证正确性
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # 性能分析（可选）
    if PROFILE:
        profile_operation("PyTorch", lambda: add(c.torch_tensor(), a.torch_tensor(), b.torch_tensor()),
                         device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_add(), device, NUM_PRERUN, NUM_ITERATIONS)

    # 清理
    check_error(LIBINFINIOP.infiniopDestroyAddDescriptor(descriptor))

# 5. 主函数
if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
```

### 复杂算子测试（Attention with KV-Cache）

```python
# attention.py
def attention(q, k, v, k_cache, v_cache, pos):
    """PyTorch 参考实现：带 KV-Cache 的因果注意力"""
    type = q.dtype
    n_q_head = q.shape[0]
    n_kv_head = k.shape[0]

    # 拼接缓存
    k_cache = k_cache[:, :pos, :]
    v_cache = v_cache[:, :pos, :]
    k = torch.cat([k_cache, k], dim=1)
    v = torch.cat([v_cache, v], dim=1)

    total_seq_len = k.shape[1]
    head_dim = v.shape[-1]

    # 分组查询注意力（GQA）支持
    if n_q_head != n_kv_head:
        q = q.reshape(n_kv_head, -1, head_dim)

    # 缩放点积注意力
    attn_scores = torch.einsum("hqd,hkd->hqk", q.to(torch.float32), k.to(torch.float32))
    attn_scores = attn_scores / (head_dim**0.5)
    attn_weights = causal_softmax(attn_scores).reshape(n_kv_head, -1, total_seq_len)

    # 加权求和
    attn_output = torch.einsum("hqk,hkd->hqd", attn_weights.to(torch.float32), v.to(torch.float32))
    return attn_output.to(type).reshape(n_q_head, -1, head_dim).permute(1, 0, 2)

def test(handle, device, n_q_head, n_kv_head, seq_len, head_dim, pos,
         k_cache_buf_len, v_cache_buf_len, q_stride, k_stride, v_stride,
         k_cache_stride, v_cache_stride, dtype=InfiniDtype.F16, sync=None):
    # 创建输入张量
    out = TestTensor([seq_len, n_q_head, head_dim], None, dtype, device, mode="zeros")
    q = TestTensor([n_q_head, seq_len, head_dim], q_stride, dtype, device, scale=0.1)
    k = TestTensor([n_kv_head, seq_len, head_dim], k_stride, dtype, device, scale=0.1)
    v = TestTensor([n_kv_head, seq_len, head_dim], v_stride, dtype, device, scale=0.1)
    k_cache = TestTensor([n_kv_head, k_cache_buf_len, head_dim], k_cache_stride, dtype, device, scale=0.1)
    v_cache = TestTensor([n_kv_head, v_cache_buf_len, head_dim], v_cache_stride, dtype, device, scale=0.1)

    # PyTorch 参考结果
    ans = attention(q.torch_tensor(), k.torch_tensor(), v.torch_tensor(),
                   k_cache.torch_tensor(), v_cache.torch_tensor(), pos)

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateAttentionDescriptor(
        handle, ctypes.byref(descriptor),
        out.descriptor, q.descriptor, k.descriptor, v.descriptor,
        k_cache.descriptor, v_cache.descriptor, pos
    ))

    # ... （后续流程与 Add 类似）
```

### PagedAttention 测试（分块缓存优化）

```python
# paged_attention.py
def ref_single_query_cached_kv_attention(
    query, key_cache, value_cache, block_tables, seq_lens, scale, alibi_slopes
):
    """PyTorch 参考实现：分块缓存的注意力"""
    output = torch.empty_like(query)
    num_query_heads, num_kv_heads = query.shape[1], value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size, block_size = value_cache.shape[3], value_cache.shape[2]
    num_seqs = query.shape[0]

    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        seq_len = seq_lens[i].item()
        block_table = block_tables[i]

        keys_lst, values_lst = [], []
        for j in range(seq_len):
            block_num = block_table[j // block_size].item()
            block_off = j % block_size
            k = key_cache[block_num, :, block_off, :]
            v = value_cache[block_num, :, block_off, :]
            keys_lst.append(k)
            values_lst.append(v)

        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)

        # GQA 支持
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        # ALiBi 偏置
        alibi_bias = None
        if alibi_slopes is not None:
            pos = torch.arange(seq_len, device=query.device).int()
            alibi_bias = (pos - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        output[i] = out.view(num_query_heads, head_size)
    return output

def test(handle, device, num_seqs, num_heads, num_kv_heads, head_size,
         block_size, max_seq_len, use_alibi, dtype=InfiniDtype.F16, sync=None):
    # 创建输入张量
    scale = 1.0 / (head_size**0.5)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq

    q = TestTensor((num_seqs, num_heads, head_size), None, dtype, device)
    out = TestTensor((num_seqs, num_heads, head_size), None, dtype, device)
    k_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    v_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)

    seq_lens_torch = torch.randint(1, max_seq_len, (num_seqs,), dtype=torch.int64)
    seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I64, device)

    block_tables_py = torch.arange(0, num_seqs * max_blocks_per_seq, dtype=torch.int64).view(num_seqs, max_blocks_per_seq)
    block_tables = TestTensor.from_torch(block_tables_py, InfiniDtype.I64, device)

    # ALiBi 斜率（可选）
    alibi_slopes_desc = ctypes.c_void_p(0)
    alibi_slopes_data = ctypes.c_void_p(0)
    alibi_slopes_torch = None
    if use_alibi:
        alibi_slopes = TestTensor((num_heads,), None, InfiniDtype.F32, device)
        alibi_slopes_desc = alibi_slopes.descriptor
        alibi_slopes_data = alibi_slopes.data()
        alibi_slopes_torch = alibi_slopes.torch_tensor()

    # PyTorch 参考结果
    ans = ref_single_query_cached_kv_attention(
        q.torch_tensor(), k_cache.torch_tensor(), v_cache.torch_tensor(),
        block_tables.torch_tensor(), seq_lens.torch_tensor(), scale, alibi_slopes_torch
    )

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreatePagedAttentionDescriptor(
        handle, ctypes.byref(descriptor),
        out.descriptor, q.descriptor, k_cache.descriptor, v_cache.descriptor,
        block_tables.descriptor, seq_lens.descriptor, alibi_slopes_desc, scale
    ))

    # ... （后续流程类似）
```

## 5. 实现细节

### 内存管理
- **策略**: TestTensor 使用 PyTorch 张量作为底层存储，通过 data_ptr() 暴露给 C 侧
- **生命周期**: Python 管理对象生命周期，C 侧仅持有原始指针，不负责释放
- **工作空间**: TestWorkspace 动态分配零大小优化，避免无效内存分配
- **描述符管理**: CTensor.destroy_desc() 显式释放 C 侧 descriptor，防止内存泄漏
- **非连续内存**: rearrange_tensor() 使用 index_add_ 实现任意步长布局

### 并发
- **设备隔离**: 每个 device 独立测试进程，通过 infinirtSetDevice 切换
- **同步点**: get_sync_func() 返回设备同步函数，GPU/NPU/MLU 需要显式 synchronize()
- **流支持**: 算子接口可选 stream 参数（当前测试未使用）

### 性能
- **预热**: profile_operation 默认 10 次 prerun，避免冷启动影响
- **迭代**: 默认 1000 次计时循环，计算平均执行时间
- **同步开销**: 每次 timed_op 包含 synchronize_device()，测量端到端延迟
- **数据类型优化**: BF16 在设备上运行，调试时转 FP32 验证

### 错误处理
- **错误码检查**: check_error(status) 抛出异常，非零返回码立即失败
- **异常安全**: test_operator 使用 try-finally 确保句柄销毁
- **跳过机制**: 不支持的测试（如广播输出、步长不匹配）提前 return
- **详细诊断**: debug() 输出不匹配元素的索引/值/期望/差异，彩色高亮

### 依赖
- **外部依赖**:
  - PyTorch: 参考实现、张量操作
  - NumPy: 二进制文件加载（from_binary）
  - ctypes: CTypes FFI 绑定
- **库依赖**:
  - libinfiniop.so: 算子内核实现
  - libinfinirt.so: 运行时设备管理
- **硬件后端**:
  - NVIDIA GPU: PyTorch CUDA
  - Ascend NPU: torch_npu
  - Cambricon MLU: torch_mlu
  - Moore Threads: torch_musa
  - Kunlun: torch_xmlir

### 设计模式
- **装饰器模式**: OpRegister.operator 自动注册函数签名
- **工厂模式**: TestTensor.from_torch(), TestTensor.from_binary()
- **策略模式**: get_sync_func(), get_tolerance() 根据设备/类型选择策略
- **RAII**: CTensor.destroy_desc() 自动资源管理
- **模板方法**: test_operator() 定义测试流程，test_func() 实现具体逻辑
- **适配器模式**: InfiniLib 统一双库接口
- **构建器模式**: 算子描述符创建（Create -> GetWorkspaceSize -> Execute -> Destroy）
