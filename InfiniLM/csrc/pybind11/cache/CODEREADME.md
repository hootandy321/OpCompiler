# Pybind11 Cache 绑定层实现文档

本模块实现了 InfiniLM KV 缓存系统的 Python 绑定层，通过 pybind11 将 C++ 的 KV 缓存配置类暴露给 Python 接口，支持静态 KV 缓存和分页 KV 缓存两种配置模式的 Python 对象封装。

## 1. 模块结构

- **`cache.hpp`**: Pybind11 绑定实现文件，定义 CacheConfig、StaticKVCacheConfig 和 PagedKVCacheConfig 的 Python 接口

## 2. 核心类

### `bind_cache` 函数
- **位置**: `cache.hpp`
- **类型**: 自由函数
- **主要功能**: 注册所有缓存配置相关的 Python 绑定，将 C++ 类暴露给 Python
- **参数**:
  - `py::module &m`: pybind11 模块引用，用于绑定类定义
- **核心逻辑**:
  - 使用 pybind11 的 `class_` 模板定义 Python 类
  - 采用智能指针 (`std::shared_ptr`) 持有策略管理 Python 对象生命周期
  - 定义构造函数、属性访问器和字符串表示方法

### `CacheConfig` Python 绑定
- **C++ 类型**: `infinilm::cache::CacheConfig`
- **Python 类型**: `CacheConfig`
- **持有策略**: `std::shared_ptr<infinilm::cache::CacheConfig>`
- **角色**: 抽象基类绑定，不直接实例化，用于类型系统
- **接口**:
  - `__repr__()`: 返回 `"<CacheConfig (abstract)>"`，标识为抽象类型

### `StaticKVCacheConfig` Python 绑定
- **C++ 类型**: `infinilm::cache::StaticKVCacheConfig`
- **Python 类型**: `StaticKVCacheConfig`
- **继承关系**: 继承自 `CacheConfig`
- **持有策略**: `std::shared_ptr<infinilm::cache::StaticKVCacheConfig>`
- **主要功能**: 静态 KV 缓存配置类，用于预分配固定大小的 KV 缓存
- **构造函数**:
  ```python
  StaticKVCacheConfig(
      max_batch_size: int = 1,
      max_cache_len: int = 2^64-1  # SIZE_T_MAX
  )
  ```
  - `max_batch_size`: 批处理大小上限，默认为 1
  - `max_cache_len`: 缓存序列长度上限，默认为 `Size` 类型的最大值
- **属性访问器**:
  - `max_batch_size() -> int`: 获取批处理大小上限
  - `max_cache_len() -> int`: 获取缓存长度上限
- **表示方法**:
  - `__repr__()`: 返回 `"<StaticKVCacheConfig>"`

### `PagedKVCacheConfig` Python 绑定
- **C++ 类型**: `infinilm::cache::PagedKVCacheConfig`
- **Python 类型**: `PagedKVCacheConfig`
- **继承关系**: 继承自 `CacheConfig`
- **持有策略**: `std::shared_ptr<infinilm::cache::PagedKVCacheConfig>`
- **主要功能**: 分页 KV 缓存配置类，用于动态块分配的 KV 缓存（类似 vLLM 的 PagedAttention）
- **构造函数**:
  ```python
  PagedKVCacheConfig(
      max_kv_memory_bytes: int,
      block_size: int = 16
  )
  ```
  - `max_kv_memory_bytes`: KV 缓存总内存字节数上限
  - `block_size`: 每块 token 数量，默认为 16
- **属性访问器**:
  - `max_kv_memory_bytes() -> int`: 获取 KV 内存上限
  - `block_size() -> int`: 获取块大小
- **表示方法**:
  - `__repr__()`: 返回 `"<PagedKVCacheConfig>"`

## 3. API 接口

### Python 接口定义

```python
# 抽象配置类（不直接使用）
class CacheConfig:
    """抽象基类，用于类型标注和多态"""
    def __repr__(self) -> str:
        return "<CacheConfig (abstract)>"

# 静态 KV 缓存配置
class StaticKVCacheConfig(CacheConfig):
    """
    静态 KV 缓存配置，预分配固定大小的缓存空间

    适用场景：
    - 批处理大小固定
    - 序列长度上限可预测
    - 内存预算充足
    """
    def __init__(
        self,
        max_batch_size: int = 1,
        max_cache_len: int = 2**64 - 1
    ) -> None:
        """
        Args:
            max_batch_size: 批处理大小上限
            max_cache_len: 缓存序列长度上限（默认为 Size 最大值）
        """

    @property
    def max_batch_size(self) -> int:
        """获取批处理大小上限"""

    @property
    def max_cache_len(self) -> int:
        """获取缓存长度上限"""

    def __repr__(self) -> str:
        return "<StaticKVCacheConfig>"

# 分页 KV 缓存配置
class PagedKVCacheConfig(CacheConfig):
    """
    分页 KV 缓存配置，动态分配块级缓存

    适用场景：
    - 批处理大小动态变化
    - 序列长度不可预测
    - 需要高效内存利用（类似 vLLM）
    """
    def __init__(
        self,
        max_kv_memory_bytes: int,
        block_size: int = 16
    ) -> None:
        """
        Args:
            max_kv_memory_bytes: KV 缓存总内存字节数
            block_size: 每块 token 数量（默认 16）
        """

    @property
    def max_kv_memory_bytes(self) -> int:
        """获取 KV 内存上限（字节）"""

    @property
    def block_size(self) -> int:
        """获取块大小（token 数）"""

    def __repr__(self) -> str:
        return "<PagedKVCacheConfig>"
```

## 4. 使用示例

### 示例 1: 创建静态 KV 缓存配置

```python
import infinilm

# 默认配置（批处理大小=1，缓存长度=无限制）
config = infinilm.StaticKVCacheConfig()

# 自定义配置
config = infinilm.StaticKVCacheConfig(
    max_batch_size=32,    # 最多支持 32 个并发请求
    max_cache_len=8192    # 每个请求最多缓存 8192 个 token
)

print(config.max_batch_size())  # 输出: 32
print(config.max_cache_len())   # 输出: 8192
```

### 示例 2: 创建分页 KV 缓存配置

```python
import infinilm

# 分页配置（2GB KV 内存，块大小 16）
config = infinilm.PagedKVCacheConfig(
    max_kv_memory_bytes=2 * 1024 * 1024 * 1024,  # 2GB
    block_size=16                                # 每块 16 个 token
)

# 默认块大小 16
config = infinilm.PagedKVCacheConfig(
    max_kv_memory_bytes=4 * 1024 * 1024 * 1024  # 4GB
)

print(config.max_kv_memory_bytes())  # 输出: 4294967296
print(config.block_size())           # 输出: 16
```

### 示例 3: 类型标注与多态

```python
from typing import Union
import infinilm

def create_cache_config(
    mode: str,
    **kwargs
) -> Union[infinilm.StaticKVCacheConfig, infinilm.PagedKVCacheConfig]:
    """
    工厂函数：根据模式创建缓存配置
    """
    if mode == "static":
        return infinilm.StaticKVCacheConfig(
            max_batch_size=kwargs.get("batch_size", 1),
            max_cache_len=kwargs.get("cache_len", 4096)
        )
    elif mode == "paged":
        return infinilm.PagedKVCacheConfig(
            max_kv_memory_bytes=kwargs["memory_bytes"],
            block_size=kwargs.get("block_size", 16)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

# 使用工厂函数
static_config = create_cache_config("static", batch_size=16, cache_len=2048)
paged_config = create_cache_config("paged", memory_bytes=8*1024**3)
```

## 5. 实现细节

### 内存管理
- **智能指针持有**: 所有配置类使用 `std::shared_ptr` 持有策略
  - Python 对象和 C++ 对象共享所有权
  - 引用计数管理生命周期
  - 避免悬垂指针和内存泄漏
- **Copy 语义**: 通过 `CacheConfig::unique_copy()` 虚函数支持深拷贝
  - 子类必须实现 `unique_copy()` 方法
  - 返回 `std::unique_ptr<CacheConfig>` 实现类型擦除

### 绑定技术
- **pybind11 框架**: 使用现代 C++ Python 绑定库
  - 比 Boost.Python 更轻量级
  - 支持自动类型转换（如 `py::arg` 默认参数）
  - 内置 STL 容器支持（`pybind11/stl.h`）
- **继承绑定**:
  - 使用 `py::class_<Derived, Base, Holder>` 模板参数
  - 支持向上转型（Derived → Base）
  - Python 端可使用 `isinstance()` 检查类型
- **默认参数**:
  - `py::arg("name") = value` 语法支持关键字参数
  - Python 端可灵活调用（位置参数或关键字参数）

### 设计模式
- **抽象工厂模式**: `CacheConfig` 作为抽象基类，`StaticKVCacheConfig` 和 `PagedKVCacheConfig` 作为具体产品
- **桥接模式**: Python 绑定层作为桥接，将 C++ 实现暴露给 Python
- **不可变对象**: 配置对象构造后状态不变（无 setter 方法）

### 性能考虑
- **零拷贝**: 属性访问器直接返回 C++ 成员变量引用或值
- **内联函数**: `__repr__` 使用 lambda 内联定义，减少函数调用开销
- **类型安全**: 编译时类型检查，运行时无类型转换开销

### 依赖关系
- **上游依赖**:
  - `../../cache/cache.hpp`: 实际的 C++ 缓存实现
  - `infinicore/tensor.hpp`: 张量类型定义
  - `pybind11/pybind11.h`: pybind11 核心库
  - `pybind11/stl.h`: STL 容器类型转换
- **命名空间**:
  - `infinilm::cache`: InfiniLM 缓存命名空间
  - `pybind11`: pybind11 绑定命名空间（别名为 `py`）

### 错误处理
- **参数验证**: 由 C++ 构造函数执行，Python 端抛出异常（如 `TypeError`）
- **生命周期保护**: shared_ptr 确保对象在 Python 使用期间不会被释放
- **虚析构函数**: `CacheConfig` 基类定义虚析构函数，支持多态删除

### 扩展性
- **新增配置类型**:
  1. 在 C++ 端实现新的 `CacheConfig` 子类
  2. 在 `bind_cache()` 函数中添加对应的 pybind11 绑定
  3. 遵循相同的接口模式（构造函数、属性、`__repr__`）
- **属性扩展**: 使用 `.def_property()` 添加 getter/setter 对
- **方法绑定**: 使用 `.def("method_name", &Class::method)` 绑定成员方法

### 编译与链接
- **头文件包含**:
  - 必须先包含 C++ 实现头文件（`cache/cache.hpp`）
  - 再包含 pybind11 头文件（避免前向声明问题）
- **符号可见性**: Python 模块编译时需导出 `bind_cache` 函数
- **模块初始化**: 在主 pybind11 模块中调用 `bind_cache(m)` 注册绑定

---

**总结**: 本模块是 InfiniLM KV 缓存系统的 Python 接口层，通过 pybind11 将 C++ 配置类暴露给 Python，支持静态和分页两种 KV 缓存模式。绑定层采用智能指针管理内存，提供类型安全的 Python API，是 C++ 引擎和 Python 应用之间的关键桥梁。
