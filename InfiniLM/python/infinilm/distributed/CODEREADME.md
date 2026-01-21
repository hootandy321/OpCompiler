# `Distributed Configuration` Core Implementation Documentation

该模块实现了 InfiniLM 框架中分布式模型训练和推理的配置管理接口。它作为 Python 前端包装器，封装了底层 C++ 实现（`_infinilm`）的分布式配置功能，主要支持张量并行（Tensor Parallelism, TP）的设备分配和规模配置。

## 1. Module Structure

- **`__init__.py`**: 模块入口文件，导出 `DistConfig` 作为公共 API
- **`dist_config.py`**: 核心实现文件，包含 `DistConfig` 类的完整实现，提供 Python 到 C++ 绑定的桥接层

## 2. Core Classes

### `DistConfig`
- **Location**: `dist_config.py`
- **Primary Function**: 分布式训练/推理配置的 Python 包装器，管理张量并行的设备拓扑和并行规模。该类通过属性代理模式，将对 Python 实例的所有操作转发到底层 C++ `_infinilm.DistConfig` 对象。
- **Key Members**:
  - `_underlying`: 底层 C++ `DistConfig` 对象的引用，实际存储所有配置状态和数据
- **Core Methods**:
  - `__init__(tp_size=None, tp_device_ids=None)`: 构造分布式配置对象。支持三种初始化模式：按并行规模自动分配（tp_size）、手动指定设备列表（tp_device_ids）、默认单设备（无参数）。使用互斥参数验证防止冲突，时间复杂度 O(1)。
  - `tp_device_ids` (property): 获取当前张量并行使用的设备 ID 列表，直接返回底层 C++ 对象的属性
  - `tp_device_ids` (setter): 设置张量并行设备列表，接受可迭代对象并转换为 Python list 后传递给 C++ 层
  - `__repr__()`: 返回对象的开发者友好的字符串表示，委托给底层 C++ 对象
  - `__str__()`: 返回对象用户友好的字符串表示，委托给底层 C++ 对象
- **Lifecycle**:
  1. **构造阶段**: 根据参数类型（tp_size 或 tp_device_ids）选择合适的 C++ 构造重载
  2. **初始化验证**: 互斥参数检查，确保 tp_size 和 tp_device_ids 不同时提供
  3. **代理模式**: 所有后续操作通过 `_underlying` 属性转发到 C++ 实现
  4. **销毁**: Python GC 自动管理，C++ 对象生命周期由 Python 对象引用计数控制

## 3. API Interface

```python
class DistConfig:
    """
    分布式模型配置类，支持张量并行设备管理。

    典型使用场景：
    - 多 GPU 训练配置
    - 张量并行的设备拓扑定义
    - 自动或手动设备分配策略
    """

    def __init__(self, tp_size: int = None, tp_device_ids: List[int] = None):
        """
        初始化分布式配置对象。

        参数:
            tp_size: 张量并行的 GPU 数量（自动分配设备）
            tp_device_ids: 显式指定使用的 GPU ID 列表

        异常:
            ValueError: 当 tp_size 和 tp_device_ids 同时提供时抛出

        注意:
            两个参数互斥，只能提供一个，或都不提供（使用默认配置）
        """

    @property
    def tp_device_ids(self) -> List[int]:
        """
        获取当前张量并行设备 ID 列表。

        返回:
            List[int]: GPU ID 列表，如 [0, 1, 2, 3]
        """

    @tp_device_ids.setter
    def tp_device_ids(self, value: Iterable[int]):
        """
        设置张量并行设备 ID 列表。

        参数:
            value: 可迭代的设备 ID 集合（如 list, tuple）

        实现细节:
            自动转换为 list 类型后传递给 C++ 层
        """
```

## 4. Usage Example

```python
from infinilm.distributed import DistConfig

# 场景 1: 自动分配 4 个 GPU 进行张量并行
config_auto = DistConfig(tp_size=4)
print(f"自动分配设备: {config_auto.tp_device_ids}")  # 输出: [0, 1, 2, 3]

# 场景 2: 手动指定特定的 GPU 设备（跳过 GPU 0，使用 GPU 1-4）
config_manual = DistConfig(tp_device_ids=[1, 2, 3, 4])
print(f"手动指定设备: {config_manual.tp_device_ids}")  # 输出: [1, 2, 3, 4]

# 场景 3: 使用默认配置（单设备/当前设备）
config_default = DistConfig()
print(f"默认配置: {config_default.tp_device_ids}")  # 输出: [0]

# 场景 4: 运行时动态修改设备分配
config = DistConfig(tp_size=2)
print(f"初始: {config.tp_device_ids}")  # [0, 1]

# 扩展到 4 个设备
config.tp_device_ids = [0, 1, 2, 3]
print(f"扩展后: {config.tp_device_ids}")  # [0, 1, 2, 3]

# 场景 5: 错误处理（同时提供两个参数）
try:
    invalid_config = DistConfig(tp_size=2, tp_device_ids=[0, 1])
except ValueError as e:
    print(f"捕获错误: {e}")  # "Provide either tp_size OR tp_device_ids, not both"
```

## 5. Implementation Details

- **语言绑定架构**:
  - 使用 pybind11 或类似技术实现 Python 到 C++ 的无缝绑定
  - 底层 `_infinilm.DistConfig` 为 C++ 实现，本模块提供 Pythonic 的接口包装
  - 属性访问通过 `@property` 装饰器实现透明代理

- **参数验证策略**:
  - 构造时执行互斥性检查（tp_size 和 tp_device_ids 不能同时存在）
  - 使用提前返回（early-return）模式：先检查冲突，再根据参数类型分发
  - 验证失败抛出 `ValueError`，提供清晰错误信息

- **数据流设计**:
  - **设置路径**: Python 对象 → list 转换 → C++ 边界 → C++ 对象内部存储
  - **获取路径**: C++ 对象属性 → Python 对象 → 返回给用户
  - 所有状态实际存储在 C++ 层，Python 层只做接口适配和类型转换

- **类型转换契约**:
  - setter 强制将输入可迭代对象转换为 `list` 类型，确保 C++ 层接收到标准 Python list
  - property 直接返回 C++ 层的属性，依赖底层绑定的类型转换机制

- **错误处理**:
  - 仅在构造时验证参数合法性
  - 属性设置不执行额外验证（委托给 C++ 层）
  - 无运行时状态检查（如设备 ID 是否有效）

- **依赖关系**:
  - **强依赖**: `infinilm.lib._infinilm` C++ 扩展模块（运行时动态导入）
  - **依赖时机**: 构造函数首次调用时导入（懒加载策略）
  - 如果 C++ 模块不可用，会在 `__init__` 时抛出 `ImportError`

- **设计模式**:
  - **代理模式（Proxy Pattern）**: `DistConfig` 作为 C++ 对象的 Python 代理
  - **外观模式（Facade Pattern）**: 隐藏 C++ 复杂性，提供简洁 Python 接口
  - **不可变语义**: 除 `tp_device_ids` setter 外，其他操作不直接修改内部状态（委托给 C++）

- **性能特性**:
  - 属性访问为 O(1) 操作，直接转发到 C++ 层
  - 构造时一次性初始化，无延迟加载
  - 无 Python 层缓存，每次属性访问都穿透到 C++ 层

- **线程安全性**:
  - Python 层无显式锁机制
  - 线程安全完全依赖底层 C++ 实现的保证
  - 多线程环境下并发设置 `tp_device_ids` 可能导致竞态条件

- **扩展性考虑**:
  - 当前仅支持 `tp_device_ids` 的读写接口
  - 未来可添加其他分布式参数（如 dp_size, pp_size 等）
  - 互斥参数验证逻辑可扩展到多参数场景
