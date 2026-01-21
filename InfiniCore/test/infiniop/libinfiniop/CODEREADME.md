# `libinfiniop` Python FFI Bindings Core Implementation Documentation

该模块提供了 libinfiniop C 库的 Python FFI (Foreign Function Interface) 绑定层，通过 ctypes 封装底层 C API，实现了张量描述符管理、算子库动态加载、多设备支持以及完整的测试工具链。这是 Infini 框架中连接 Python 测试代码与 C++ 运算库的核心桥梁层。

## 1. Module Structure

- **`__init__.py`**: 模块入口，导出核心类和全局单例 `LIBINFINIOP`
- **`datatypes.py`**: 定义 22 种 Infini 数据类型枚举及其名称映射
- **`devices.py`**: 定义 10 种硬件设备枚举及与 PyTorch 设备字符串的双向映射
- **`structs.py`**: 使用 ctypes 定义底层 C 结构体（TensorDescriptor、Handle、OpDescriptor）的 Python 表示
- **`liboperators.py`**: 实现动态库加载器 `open_lib()` 和库合并代理类 `InfiniLib`
- **`op_register.py`**: 算子注册中心，定义 40+ 个算子的 ctypes 函数签名规范
- **`utils.py`**: 测试工具核心实现，包含张量封装类 TestTensor、验证函数 debug、性能分析工具

## 2. Core Classes

### `InfiniLib`
- **Location**: `liboperators.py:14-26`
- **Primary Function**: 合并代理类，将 libinfiniop.so 和 libinfinirt.so 两个动态库的接口统一为一个访问点
- **Key Members**:
  - `librt`: CDLL 实例，运行时库句柄（libinfinirt.so）
  - `libop`: CDLL 实例，算子库句柄（libinfiniop.so）
- **Core Methods**:
  - `__getattr__(name)`: 智能属性查找，优先在 libop 中查找，失败则查找 librt，均失败抛 AttributeError
- **Lifecycle**: 由 `open_lib()` 函数构造并返回全局单例 `LIBINFINIOP`

### `OpRegister`
- **Location**: `op_register.py:10-22`
- **Primary Function**: 算子注册中心，使用装饰器模式收集所有算子的 ctypes 签名配置函数
- **Key Members**:
  - `registry`: 类变量，存储所有被 `@OpRegister.operator` 装饰的函数对象
- **Core Methods**:
  - `operator(op)`: 装饰器工厂，将算子配置函数注册到 registry 列表
  - `register_lib(lib)`: 遍历 registry 并逐个调用，配置 CDLL 实例的函数签名（argtypes/restype）
- **Design Pattern**: Registry Pattern，延迟初始化（在 open_lib() 时才触发配置）

### `TestTensor` (extends `CTensor`)
- **Location**: `utils.py:43-188`
- **Primary Function**: 测试用张量类，封装 PyTorch 张量并提供与底层 C API 的互操作性，支持多种初始化模式和自定义步长
- **Key Members**:
  - `descriptor`: `infiniopTensorDescriptor_t`，底层 C 张量描述符
  - `dt`: `InfiniDtype`，Infini 数据类型枚举值
  - `device`: `InfiniDeviceEnum`，目标设备类型
  - `shape`: list[int]，张量形状
  - `strides`: list[int] | None，自定义步长（用于广播、转置等场景）
  - `_torch_tensor`: torch.Tensor，逻辑张量（标准形状）
  - `_data_tensor`: torch.Tensor，物理张量（自定义步长后的实际存储布局）
- **Core Methods**:
  - `__init__(shape, strides, dt, device, mode, scale, bias, set_tensor, randint_low, randint_high)`: 构造函数，支持 6 种初始化模式：
    - `"random"`: 随机浮点数或随机整数（根据类型判断）
    - `"zeros"`: 全零张量
    - `"ones"`: 全一张量
    - `"randint"`: 均匀分布随机整数
    - `"float8_e4m3fn"`: FP8 E4M3 格式（用于量化测试）
    - `"manual"` / `"binary"`: 从已有 torch.Tensor 或二进制文件加载
  - `torch_tensor()`: 返回逻辑张量（用于计算期望值）
  - `actual_tensor()`: 返回物理张量（用于传入 C 算子）
  - `data()`: 返回底层设备指针（data_ptr()）
  - `is_broadcast()`: 检查是否为广播张量（步长中含 0）
  - `from_binary(binary_file, shape, strides, dt, device)`: 类方法，从 numpy 二进制文件加载
  - `from_torch(torch_tensor, dt, device)`: 类方法，从 PyTorch 张量转换
  - `update_torch_tensor(new_tensor)`: 更新逻辑张量（用于就地操作后同步）
- **Key Algorithm** - 步长计算：
  ```python
  if strides is None:
      strides = [1 for _ in shape]
      for i in range(self.ndim - 2, -1, -1):
          strides[i] = strides[i + 1] * shape[i + 1]  # C-order row-major
  ```
- **Memory Layout**: 物理张量通过 `rearrange_tensor()` 函数根据目标步长重新排列内存布局，使用 torch.index_add_ 实现稀疏索引填充

### `CTensor`
- **Location**: `utils.py:15-41`
- **Primary Function**: 基础张量类，负责创建和管理底层 C 张量描述符
- **Key Members**:
  - `descriptor`: `infiniopTensorDescriptor_t`，指向 C 结构体的指针
  - `ndim`: int，张量维度数
  - `c_shape`: ctypes 数组，C 兼容的形状数组
  - `c_strides`: ctypes 数组，C 兼容的步长数组
- **Core Methods**:
  - `__init__(dt, shape, strides)`: 调用 `LIBINFINIOP.infiniopCreateTensorDescriptor()` 创建描述符
  - `destroy_desc()`: 调用 `LIBINFINIOP.infiniopDestroyTensorDescriptor()` 释放资源

### `TestWorkspace`
- **Location**: `utils.py:259-274`
- **Primary Function**: 算子工作空间管理器，为需要额外临时内存的算子（如 attention、matmul）提供缓冲区
- **Key Members**:
  - `tensor`: TestTensor | None，指向 U8 类型的工作空间张量
  - `_size`: int，工作空间字节数
- **Core Methods**:
  - `__init__(size, device)`: 创建全一 U8 张量作为缓冲区（size=0 时为 None）
  - `data()`: 返回设备指针或 None
  - `size()`: 返回 ctypes.c_uint64 包装的大小值

### `TensorDescriptor` (ctypes Structure)
- **Location**: `structs.py:4-5`
- **Primary Function**: 映射 C 端不透明结构体 `infiniopTensorDescriptor`
- **Structure Definition**:
  ```python
  _fields_ = []  # 不透明结构体，无需定义字段
  ```
- **Usage**: 通过 `POINTER(TensorDescriptor)` 定义 `infiniopTensorDescriptor_t` 类型，仅用于类型标注和传递指针

### `Handle` (ctypes Structure)
- **Location**: `structs.py:11-12`
- **Primary Function**: 映射 C 端设备句柄结构体
- **Structure Definition**:
  ```python
  _fields_ = [("device", c_int), ("device_id", c_int)]
  ```
- **Fields**:
  - `device`: 设备类型枚举值（如 InfiniDeviceEnum.NVIDIA = 1）
  - `device_id`: 设备 ID（通常为 0，表示第一张卡）

### `OpDescriptor` (ctypes Structure)
- **Location**: `structs.py:18-19`
- **Primary Function**: 映射 C 端算子描述符结构体
- **Structure Definition**:
  ```python
  _fields_ = [("device", c_int), ("device_id", c_int)]
  ```
- **Usage**: 每个算子在创建描述符时返回该结构体的指针，用于存储算子特定的元数据和配置

## 3. API Interface

### 库加载与初始化

```python
def open_lib() -> InfiniLib:
    """
    动态加载 libinfiniop.so 和 libinfinirt.so，配置 ctypes 函数签名，
    注册所有算子，返回合并后的库代理对象。

    环境变量：
        INFINI_ROOT: 库文件安装根目录，默认为 ~/.infini

    返回：
        InfiniLib: 统一的库接口代理对象

    异常：
        AssertionError: 找不到 .so 或 .dll 文件

    副作用：
        设置全局单例 LIBINFINIOP
        调用 OpRegister.register_lib(lib)
    """
```

### 张量描述符管理

```python
# 创建张量描述符（通过 LIBINFINIOP 调用）
LIBINFINIOP.infiniopCreateTensorDescriptor(
    POINTER(infiniopTensorDescriptor_t),  # out: 描述符指针
    c_uint64,                             # ndim: 维度数
    POINTER(c_uint64),                    # shape: 形状数组
    POINTER(c_int64),                     # strides: 步长数组
    c_int,                                # dtype: 数据类型枚举
) -> c_int  # 返回状态码（0 表示成功）

# 销毁张量描述符
LIBINFINIOP.infiniopDestroyTensorDescriptor(
    infiniopTensorDescriptor_t,  # 描述符指针
) -> c_int
```

### 设备句柄管理

```python
def create_handle() -> infiniopHandle_t:
    """
    创建 Infini 运行时句柄，用于后续算子创建。

    返回：
        infiniopHandle_t: 句柄指针
    """

def destroy_handle(handle: infiniopHandle_t) -> None:
    """
    销毁运行时句柄，释放资源。

    参数：
        handle: 由 create_handle() 返回的句柄
    """

# 设置当前设备（调用 C 运行时 API）
LIBINFINIOP.infinirtSetDevice(c_int, c_int) -> c_int
```

### 工具函数

```python
def check_error(status: int) -> None:
    """
    检查 C 函数返回的状态码，非零时抛出异常。

    参数：
        status: C 函数返回值

    异常：
        Exception: 当 status != 0 时
    """

def to_torch_dtype(dt: InfiniDtype, compatability_mode: bool = False) -> torch.dtype:
    """
    将 InfiniDtype 枚举映射到 PyTorch dtype。

    支持类型：
        BOOL, BYTE, I8-64, U8-64, F16, F32, F64, BF16, F8 (float8_e4m3fn)

    兼容性模式：
        older PyTorch 版本可能不支持某些无符号类型，fallback 到有符号类型
    """

def to_numpy_dtype(dt: InfiniDtype, compatability_mode: bool = False) -> np.dtype:
    """
    将 InfiniDtype 映射到 NumPy dtype，用于二进制文件加载。
    """

def rearrange_tensor(tensor: torch.Tensor, new_strides: list[int]) -> torch.Tensor:
    """
    根据指定步长重新排列张量的物理内存布局。

    算法：
        1. 计算每个维度在新步长下的索引偏移
        2. 使用 torch.meshgrid 生成原始索引网格
        3. 计算线性化后的新位置索引
        4. 使用 index_add_ 将原始数据稀疏填充到新张量

    限制：
        暂不支持负步长（会抛 ValueError）

    复杂度：
        O(N) where N = tensor.numel()
    """

def debug(actual: torch.Tensor, desired: torch.Tensor,
          atol: float = 0, rtol: float = 1e-2,
          equal_nan: bool = False, verbose: bool = True) -> None:
    """
    比对两个张量并打印详细差异报告。

    功能：
        - 自动将 BF16 转为 FP32 进行比对
        - 计算绝对误差和相对误差掩码
        - 打印不匹配元素的索引、实际值、期望值、差值
        - 输出统计信息（dtype、容差、错误率、最值）

    格式化输出：
        使用 ANSI 颜色代码（31红/32绿/33黄）高亮差异
    """

def test_operator(device: InfiniDeviceEnum,
                  test_func: Callable,
                  test_cases: list[tuple],
                  tensor_dtypes: list[torch.dtype]) -> None:
    """
    运算测试框架主入口，遍历所有测试用例和数据类型。

    流程：
        1. 调用 infinirtSetDevice 设置设备
        2. 创建运行时句柄
        3. 根据设备过滤不支持的 dtype（如某些 GPU 不支持 BF16）
        4. 遍历 test_cases 和 tensor_dtypes，逐个调用 test_func
        5. 在 finally 中销毁句柄

    典型 test_func 签名：
        (handle, device, *test_case_args, tensor_dtype, sync_func)
    """

def get_test_devices(args: argparse.Namespace) -> list[InfiniDeviceEnum]:
    """
    根据命令行参数解析要测试的设备列表。

    支持参数：
        --cpu, --nvidia, --cambricon, --ascend, --metax, --moore,
        --kunlun, --hygon, --iluvatar, --qy

    特殊处理：
        - Ascend NPU 需要显式调用 torch.npu.set_device(0)
        - MooreThreads 需要 torch_musa
        - Kunlun 需要 torch_xmlir
        - 默认返回 [CPU]
    """

def profile_operation(desc: str, func: Callable, torch_device: str,
                      NUM_PRERUN: int, NUM_ITERATIONS: int) -> None:
    """
    统一性能分析工作流。

    步骤：
        1. 执行 NUM_PRERUN 次 warmup
        2. 执行 NUM_ITERATIONS 次计时运行
        3. 每次迭代前后调用 synchronize_device() 确保异步操作完成
        4. 打印平均耗时（毫秒）

    示例输出：
        "Matmul time: 1.234567 ms"
    """

def timed_op(func: Callable, num_iterations: int, device: str) -> float:
    """
    带同步的计时函数，返回单次迭代平均耗时（秒）。
    """
```

## 4. Usage Example

```python
# ============================================================
# 示例：测试矩阵乘法算子 (GEMM)
# ============================================================

import sys
sys.path.insert(0, "/path/to/InfiniCore/test/infiniop")

from libinfiniop import *
from libinfiniop.utils import *

# 1. 获取命令行参数
args = get_args()
devices = get_test_devices(args)  # 如 [InfiniDeviceEnum.NVIDIA]

# 2. 定义测试函数
def test_gemm(handle, device, m, n, k, tensor_dtype, sync_func):
    # 创建测试张量
    A = TestTensor((m, k), None, InfiniDtype.F32, device, mode="random")
    B = TestTensor((k, n), None, InfiniDtype.F32, device, mode="random")
    C = TestTensor((m, n), None, InfiniDtype.F32, device, mode="zeros")

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateGemmDescriptor(
        handle,
        ctypes.byref(descriptor),
        A.descriptor,
        B.descriptor,
        C.descriptor,
    ))

    # 查询工作空间大小
    workspace_size = ctypes.c_size_t()
    check_error(LIBINFINIOP.infiniopGetGemmWorkspaceSize(
        descriptor, ctypes.byref(workspace_size)
    ))

    # 分配工作空间
    workspace = TestWorkspace(workspace_size.value, device)

    # 调用算子
    check_error(LIBINFINIOP.infiniopGemm(
        descriptor,
        workspace.data(),
        workspace.size(),
        C.data(),
        A.data(),
        B.data(),
        ctypes.c_float(1.0),  # alpha
        ctypes.c_float(0.0),  # beta
        None,  # stream
    ))

    # 同步设备
    if sync_func:
        sync_func()

    # 验证结果
    expected = torch.matmul(A.torch_tensor(), B.torch_tensor())
    if args.debug:
        debug(C.actual_tensor(), expected, atol=1e-3, rtol=1e-2)
    else:
        torch.testing.assert_close(C.actual_tensor(), expected,
                                   rtol=1e-2, atol=1e-3)

    # 清理
    LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor)

# 3. 定义测试用例
test_cases = [
    (128, 256, 512),   # m=128, n=256, k=512
    (1024, 1024, 1024),
    (64, 128, 256),
]

# 4. 运行测试
for device in devices:
    test_operator(device, test_gemm, test_cases, [torch.float32])

# ============================================================
# 示例：带自定义步长的张量测试（广播场景）
# ============================================================

def test_broadcast_add():
    handle = create_handle()
    device = InfiniDeviceEnum.CPU

    # 形状 (3, 1) 广播到 (3, 4)
    A = TestTensor((3, 1), [4, 1], InfiniDtype.F32, device, mode="random")
    B = TestTensor((3, 4), None, InfiniDtype.F32, device, mode="random")
    C = TestTensor((3, 4), None, InfiniDtype.F32, device, mode="zeros")

    # A.is_broadcast() == True（步长含 0）

    # ... 调用 Add 算子 ...

    # 验证时使用 torch_tensor()（逻辑张量）
    expected = A.torch_tensor() + B.torch_tensor()
    debug(C.actual_tensor(), expected)

    destroy_handle(handle)

# ============================================================
# 示例：从二进制文件加载张量（用于回归测试）
# ============================================================

def test_from_binary():
    tensor = TestTensor.from_binary(
        binary_file="testdata/matmul_input_A.bin",
        shape=(1024, 1024),
        strides=(1024, 1),  # C-order
        dt=InfiniDtype.F16,
        device=InfiniDeviceEnum.NVIDIA,
    )

    # tensor.data() 可直接传递给 C 算子
    # tensor.torch_tensor() 可用于 PyTorch 参考计算
```

## 5. Implementation Details

### Memory Management
- **策略**: 基于 RAII (Resource Acquisition Is Initialization) 模式，TestTensor 在构造时创建 C 描述符，在 `destroy_desc()` 中释放
- **工作空间**: TestTensor 持有对底层 torch.Tensor 的强引用，通过 `data_ptr()` 获取设备地址传递给 C API
- **缓冲区生命周期**: TestWorkspace 使用全一张量作为临时缓冲区，无需手动初始化，算子执行完后由 Python GC 自动回收
- **跨设备传输**: 通过 torch_tensor.to(device) 实现 CPU 与设备间的数据拷贝，不涉及手动内存分配

### Concurrency
- **设备同步**: `synchronize_device()` 根据 torch_device 字符串分发到对应的同步函数：
  - cuda: `torch.cuda.synchronize()`
  - npu: `torch.npu.synchronize()`
  - mlu: `torch.mlu.synchronize()`
  - musa: `torch.musa.synchronize()`
- **流支持**: 所有算子函数签名的最后一个参数为 `c_void_p stream`，当前实现传 None（使用默认流）
- **无锁设计**: Python 层无显式锁，依赖 PyTorch 的设备 API 线程安全性

### Performance
- **零拷贝**: TestTensor 的 `data()` 直接返回底层 torch.Tensor 的设备指针，避免数据复制
- **延迟初始化**: 工作空间仅在算子查询到非零大小时才分配
- **Warmup 机制**: `profile_operation()` 默认执行 10 次 prerun 避免冷启动影响
- **类型转换开销**: BF16 比对时自动转为 FP32，引入额外转换成本（仅在 debug 模式）

### Error Handling
- **状态码检查**: 所有 C API 调用返回值通过 `check_error()` 校验，非零时抛异常
- **资源清理**: 使用 try-finally 确保句柄销毁：`test_operator()` 在 finally 中调用 `destroy_handle()`
- **断言验证**: 张量形状、步长一致性在构造时通过 assert 检查
- **详细日志**: `debug()` 和 `print_discrepancy()` 提供元素级差异报告，包含颜色编码和统计信息

### Dependencies
- **外部库**:
  - `ctypes`: Python 标准库，用于 C FFI 绑定
  - `torch`: PyTorch 张量计算和设备管理
  - `numpy`: 二进制文件 I/O 和数据类型转换
  - `platform`: 跨平台动态库扩展名判断（.so / .dll）
- **设备特定后端**（可选）:
  - `torch_mlu`: 寒武纪 MLU 支持
  - `torch_npu`: 华为 Ascend NPU 支持
  - `torch_musa`: 摩尔线程 MUSA 支持
  - `torch_xmlir`: 昆仑 XPU 支持
- **底层 C 库**:
  - `libinfiniop.so`: 算子实现库
  - `libinfinirt.so`: 运行时设备管理库

### Design Patterns
- **Facade Pattern**: InfiniLib 将两个动态库的接口统一为一个访问点
- **Registry Pattern**: OpRegister 使用类变量列表 + 装饰器实现算子自动注册
- **Factory Pattern**: `open_lib()` 根据平台和 INFINI_ROOT 路径动态加载库
- **Strategy Pattern**: `get_sync_func()` 根据设备类型返回不同的同步策略
- **Template Method**: `test_operator()` 定义测试流程骨架，具体测试逻辑由传入的 test_func 实现
- **Adapter Pattern**: TestTensor 适配 PyTorch 张量到 C API 接口（描述符 + 设备指针）
