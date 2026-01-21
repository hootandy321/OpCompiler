# `infinilm.cache` 核心实现文档

本模块是 InfiniLM 框架中 KV 缓存配置系统的 Python 接口层，提供了对底层 C++ 实现的类型安全包装。它定义了三种缓存配置类，支持静态 KV 缓存和分页 KV 缓存两种不同的内存管理策略，用于优化大模型推理过程中的键值存储。

## 1. 模块结构

- **`__init__.py`**: 模块公开接口导出，将核心配置类暴露给上层应用
- **`cache.py`**: 缓存配置类的 Python 包装实现，继承自 C++ 绑定类

## 2. 核心类

### `CacheConfig`
- **位置**: `cache.py:4-8`
- **主要功能**: 抽象基类，定义所有缓存配置的公共接口
- **关键成员**:
  - 继承自 `_infinilm.CacheConfig` (C++ 绑定基类)
- **核心方法**:
  - `__init__()`: 禁止实例化，抛出 `NotImplementedError` 强制使用子类
- **生命周期**: 纯抽象接口，永不实例化，仅作为类型继承契约

### `StaticKVCacheConfig`
- **位置**: `cache.py:11-13`
- **主要功能**: 静态 KV 缓存配置，为批次预分配固定大小的连续内存块
- **关键成员**:
  - 继承自 `CacheConfig` 和 `_infinilm.StaticKVCacheConfig` (多重继承)
- **核心方法**:
  - `__init__(max_batch_size: int, max_cache_len: int)`: 初始化静态缓存配置
    - `max_batch_size`: 最大批次大小（并发请求数量）
    - `max_cache_len`: 每个序列的最大缓存长度（token 数量）
    - 调用 C++ 构造函数分配固定内存池
- **生命周期**: 显式构造，由 Python 传递参数到 C++ 层进行内存分配

### `PagedKVCacheConfig`
- **位置**: `cache.py:16-26`
- **主要功能**: 分页 KV 缓存配置，使用块状内存管理实现动态内存分配和回收
- **关键成员**:
  - 继承自 `CacheConfig` 和 `_infinilm.PagedKVCacheConfig`
- **核心方法**:
  - `__init__(max_kv_memory_bytes: int, block_size: int = 16)`: 初始化分页缓存配置
    - `max_kv_memory_bytes`: KV 缓存总内存预算（字节）
    - `block_size`: 每个页块的 token 数量，默认 16（优化内存碎片和访问效率）
    - 调用 C++ 构造函数建立页块管理器
- **生命周期**: 显式构造，C++ 层维护页块空闲列表和分配映射

## 3. API 接口

```python
# 抽象基类 - 不可直接实例化
class CacheConfig(_infinilm.CacheConfig):
    def __init__(self):
        # 抛出 NotImplementedError，强制使用子类
        raise NotImplementedError("CacheConfig is an abstract class...")

# 静态 KV 缓存配置
class StaticKVCacheConfig(CacheConfig):
    def __init__(
        self,
        max_batch_size: int = 1,      # 默认单批次
        max_cache_len: int = 0        # 0 表示无限制（由 C++ 层处理）
    ) -> None:
        # 初始化固定内存池
        # 时间复杂度: O(max_batch_size * max_cache_len) 内存分配
        pass

# 分页 KV 缓存配置
class PagedKVCacheConfig(CacheConfig):
    def __init__(
        self,
        max_kv_memory_bytes: int,     # 必需参数：总内存预算
        block_size: int = 16          # 默认 16 tokens/块
    ) -> None:
        # 初始化页块分配器
        # 空间复杂度: O(max_kv_memory_bytes / block_size) 页块管理
        pass
```

## 4. 使用示例

```python
# 示例 1: 使用静态 KV 缓存（适合固定批次、已知序列长度）
from infinilm.cache import StaticKVCacheConfig

# 配置: 最多 8 个并发请求，每个请求缓存最多 4096 个 token
static_config = StaticKVCacheConfig(
    max_batch_size=8,
    max_cache_len=4096
)
# C++ 层预分配 8 * 4096 * 2 * hidden_dim 字节的连续内存

# 示例 2: 使用分页 KV 缓存（适合变长序列、动态批次）
from infinilm.cache import PagedKVCacheConfig

# 配置: 总共 2GB KV 缓存，每块 16 个 token
paged_config = PagedKVCacheConfig(
    max_kv_memory_bytes=2 * 1024 * 1024 * 1024,  # 2GB
    block_size=16                                 # 默认值通常最优
)
# C++ 层建立页块管理器，支持动态分配和回收

# 示例 3: 类型检查与多态使用
from infinilm.cache import CacheConfig, StaticKVCacheConfig, PagedKVCacheConfig

def configure_engine(config: CacheConfig) -> None:
    # 多态接口：接受任何 CacheConfig 子类
    if isinstance(config, StaticKVCacheConfig):
        print(f"使用静态缓存: batch={config.max_batch_size}")
    elif isinstance(config, PagedKVCacheConfig):
        print(f"使用分页缓存: mem={config.max_kv_memory_bytes}")

configure_engine(static_config)  # ✓ 类型安全
configure_engine(paged_config)   # ✓ 类型安全
```

## 5. 实现细节

### 内存管理策略
- **静态缓存 (StaticKVCache)**:
  - 预分配策略：在初始化时一次性分配所有需要的连续内存
  - 优点：零运行时分配开销，内存访问模式确定，缓存友好
  - 缺点：内存利用率低（需按最大需求预留），不支持超出配置的序列
  - 适用场景：离线批处理、固定长度输入、延迟敏感的在线服务

- **分页缓存 (PagedKVCache)**:
  - 动态分配策略：将内存切分为固定大小的块（block），按需分配和回收
  - 优点：内存利用率高，支持变长序列和动态批次，可实现请求间的内存共享
  - 缺点：运行时分配开销，需要页块管理元数据，可能产生内存碎片
  - 适用场景：在线推理服务、对话系统（长序列）、多租户共享 GPU

### 并发与线程安全
- **C++ 绑定层**：`_infinilm` 模块负责实际的线程安全实现
  - `CacheConfig` 对象通常是**不可变的配置对象**，创建后不应修改
  - C++ 层使用互斥锁或原子操作保护页块分配器的内部状态
- **Python 层**：本模块提供的配置类是**轻量级包装**，无额外同步机制
  - 配置对象应在模型初始化时创建，避免在推理热路径中重建

### 性能优化
- **块大小选择 (block_size)**：
  - 默认值 16 是经验最优值，平衡了以下因素：
    - 太小（如 1）：管理开销大，碎片多
    - 太大（如 128）：内存浪费，不适合短序列
  - 计算公式：`单个块大小 = block_size * 2 * hidden_dim * sizeof(float16)`
    - 对于 hidden_dim=4096，block_size=16：约 16 * 2 * 4096 * 2 = 256KB
  - 建议：保持默认值，除非有特殊工作负载特征

- **内存预算计算**：
  - 静态缓存：`内存 = max_batch_size * max_cache_len * 2 * hidden_dim * elem_size`
  - 分页缓存：需为 KV 缓存预留总内存的 30-50%，剩余给模型权重和激活值

### 错误处理
- **抽象类强制**：直接实例化 `CacheConfig` 会抛出 `NotImplementedError`
  - 这是 Python 层的类型安全保护，C++ 层可能有自己的检查
- **参数验证**：由 C++ 层负责（如 `max_batch_size > 0`, `max_kv_memory_bytes > block_size`）
  - Python 层未实现参数检查，依赖 C++ 绑定抛出异常
- **内存不足**：当超出配置的内存限制时，C++ 层会返回错误或触发 OOM

### 依赖关系
- **外部依赖**：
  - `infinilm.lib._infinilm`: Cython/pybind11 生成的 C++ 扩展模块
    - 提供 `CacheConfig`, `StaticKVCacheConfig`, `PagedKVCacheConfig` 的 C++ 实现
- **模块依赖**：
  - 被 `infinilm` 更上层模块使用（如模型初始化、引擎配置）
  - 无依赖其他 `infinilm` 子模块（独立的配置层）

### 设计模式
- **工厂模式（隐式）**：用户根据场景选择 `StaticKVCacheConfig` 或 `PagedKVCacheConfig`
- **包装器模式 (Wrapper)**：Python 类是 C++ 对象的薄包装，提供类型提示和文档
- **模板方法模式**：`CacheConfig` 定义接口契约，子类实现具体配置逻辑
- **不可变对象模式**：配置对象创建后不应修改，保证线程安全和可预测性

### 扩展性考虑
- **添加新缓存策略**：
  1. 在 C++ 层实现新的 `XXXKVCacheConfig` 类
  2. 在 Python 层添加包装类继承 `CacheConfig` 和 `_infinilm.XXXKVCacheConfig`
  3. 在 `__init__.py` 中导出新类
- **当前架构优势**：Python 和 C++ 解耦，C++ 可独立优化实现细节
