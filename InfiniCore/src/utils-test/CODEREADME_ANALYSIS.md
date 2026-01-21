# utils-test 架构全景

## 1. 子系统职责

`utils-test` 是 InfiniCore/utils 模块的**独立测试套件**，专注于验证数组重排（rearrange）操作的正确性。该模块是一个叶子节点（无子目录），提供了一套完整的单元测试框架，用于验证 utils 模块中 `rearrange` 函数在各种张量形状、步长和数据布局组合下的正确性。

在 InfiniCore 整体架构中，本目录位于测试层的最底层，直接依赖 `../utils.h` 提供的工具函数，是保证基础工具正确性的关键测试组件。

## 2. 模块导航

**重要说明**: 该目录是一个叶子节点（Leaf Node），不包含任何子目录。该目录本身是一个测试代码包，直接包含测试源文件。

**目录内容**:
- `main.cc`: 测试主入口，调用各个测试函数并汇总失败次数
- `test_rearrange.cc`: 数组重排功能的核心测试实现
- `utils_test.h`: 测试函数声明，引入 `../utils.h` 依赖

## 3. 架构逻辑图解

### 3.1 模块定位

```
InfiniCore/src/
    ├── utils/                    # 被测模块
    │   ├── rearrange.cc/h        # 数组重排实现
    │   └── ...
    └── utils-test/               # 测试模块（当前目录）
        ├── main.cc               # 测试入口
        ├── test_rearrange.cc     # 测试用例实现
        └── utils_test.h          # 测试接口声明
```

### 3.2 测试执行流程

```
测试启动: main.cc::main()
    ↓
调用: test_rearrange()
    ↓
测试场景矩阵: test_transpose_any()
    ├─ 测试用例 1: 2D 转置 (3x5)
    │   ├─ 形状: {3, 5}
    │   ├─ 步长 A: {5, 1} (行主序)
    │   └─ 步长 B: {1, 3} (列主序，转置)
    ├─ 测试用例 2: 1D 无变化 (1x2048)
    │   ├─ 形状: {1, 2048}
    │   ├─ 步长 A: {2048, 1}
    │   └─ 步长 B: {2048, 1} (相同布局)
    ├─ 测试用例 3: 4D 复杂重排 (2x2x2x4)
    │   ├─ 形状: {2, 2, 2, 4}
    │   ├─ 步长 A: {16, 8, 1, 2}
    │   └─ 步长 B: {16, 8, 4, 1} (混合维度交换)
    └─ 测试用例 4: 5D 高维重排 (2x2x2x2x4)
        ├─ 形状: {2, 2, 2, 2, 4}
        ├─ 步长 A: {32, 16, 8, 1, 2}
        └─ 步长 B: {32, 16, 8, 4, 1} (深度维度重排)
    ↓
验证逻辑: check_equal<float>()
    ├─ 初始化测试数据: a[i] = (float)i / numel
    ├─ 调用: utils::rearrange(b.data(), a.data(), ...)
    ├─ 逐元素比较: memcmp(ptr_a, ptr_b, element_size)
    ├─ 步长遍历: incrementOffset() 处理多维索引
    └─ 错误报告: std::cerr << "Error at " << i
    ↓
结果汇总: 返回失败测试数量到 main()
    └─ 返回码: 0 (全部通过) 或 >0 (失败数量)
```

### 3.3 核心测试算法

**1. 数据生成策略**:
```cpp
for (size_t i = 0; i < numel; i++) {
    a[i] = (float)i / numel;  // 归一化唯一值，便于定位错误位置
}
```

**2. 多维索引遍历算法** (`incrementOffset`):
- 从最低维（最右侧）开始递增计数器
- 通过步长（strides）计算字节偏移量
- 当计数器达到维度大小时，进位到更高维度
- 自动处理回绕，重置当前维度偏移

**3. 严格相等性检查**:
- 使用 `memcmp` 进行字节级比较
- 支持任意数据类型（通过模板 `<typename T>`）
- 精确报告每个错误元素的索引和值

### 3.4 测试覆盖维度

| 测试维度 | 覆盖范围 |
|---------|---------|
| **张量维度** | 1D 到 5D |
| **形状大小** | 最小 3 个元素，最大 64 个元素（2x2x2x2x4） |
| **布局模式** | 行主序、列主序、混合步长 |
| **数据类型** | float（可扩展到其他类型） |
| **边界条件** | 单维度（shape 包含 1）、转置（步长交换） |

### 3.5 与被测模块的交互

```
utils-test (测试代码)
    │ #include "../utils.h"
    ↓
utils::rearrange()
    ├─ 输入: void *dst, void *src, shape[], strides_dst[], strides_src[]
    ├─ 功能: 按照指定步长模式重排数组元素
    └─ 实现: ../utils/rearrange.cc
```

## 4. 技术特点

### 4.1 测试设计亮点

1. **参数化测试**: `test_transpose_any()` 函数接受任意形状和步长组合，实现高度可复用的测试逻辑
2. **自包含测试数据**: 不依赖外部文件，测试数据在运行时生成
3. **精确错误定位**: 报告错误发生的确切元素索引和预期/实际值
4. **多维步长验证**: 不仅测试数据内容，还验证复杂内存布局下的索引计算

### 4.2 测试覆盖策略

- **正向测试**: 验证各种合法形状和步长组合下的正确性
- **边界测试**: 包含单维度（size=1）的特殊情况
- **性能无关**: 当前测试关注正确性，未包含性能基准测试

### 4.3 局限性与改进方向

**当前局限**:
- 仅测试 `float` 类型，未覆盖 `int`、`half`、`double` 等其他数据类型
- 测试用例数量有限（4 个），未进行系统性覆盖（如负步长、非连续内存）
- 未包含异常场景测试（如空指针、形状不匹配）
- 未进行性能回归测试

**建议改进**:
1. 扩展数据类型覆盖（使用模板测试）
2. 添加属性测试（Property-Based Testing），随机生成形状和步长
3. 引入性能基准测试框架（如 Google Benchmark）
4. 集成到 CI/CD 流程，每次代码提交自动运行

## 5. 使用指南

### 5.1 编译测试

```bash
# 假设使用 XMake 构建系统
xmake build utils-test
```

### 5.2 运行测试

```bash
./utils-test
```

**预期输出**:
```
test_transpose 1 passed
test_transpose 2 passed
test_transpose 3 passed
test_transpose 4 passed
```

**失败输出示例**:
```
Error at 7: 0.0035 vs 0.0042
test_transpose 1 failed
```

### 5.3 添加新测试用例

在 `test_rearrange()` 函数中添加新的 `test_transpose_any()` 调用：

```cpp
int test_rearrange() {
    return test_transpose_any(1, {3, 5}, {5, 1}, {1, 3})
         + test_transpose_any(5, {4, 6, 8}, {48, 8, 1}, {1, 4, 32})  // 新测试
         + test_transpose_any(2, {1, 2048}, {2048, 1}, {2048, 1})
         // ... 其他测试
}
```

## 6. 依赖关系

### 内部依赖
- `../utils.h`: 提供 `utils::rearrange()` 函数声明

### 外部依赖
- C++ 标准库:
  - `<cstring>`: memcmp 内存比较
  - `<iostream>`: std::cerr, std::cout 输出
  - `<numeric>`: std::accumulate 计算 numel
  - `<vector>`: 动态数组容器

### 编译依赖
- C++11 或更高标准（支持 `auto` 关键字、范围 for 循环）
- 与 InfiniCore 主项目共享构建配置

## 7. 质量保证

### 测试覆盖率

| 代码路径 | 覆盖状态 |
|---------|---------|
| `utils/rearrange.cc` 核心逻辑 | ✅ 部分覆盖（通过典型用例） |
| 边界条件（空指针、零维度） | ⚠️ 未覆盖 |
| 异常路径（内存分配失败） | ⚠️ 未覆盖 |
| 不同数据类型 | ⚠️ 仅覆盖 float |

### 回归测试策略

建议在以下场景运行本测试套件：
1. 修改 `utils/rearrange.cc` 实现后
2. 升级编译器或 C++ 标准库后
3. 移植到新硬件平台后
4. 优化内存布局算法后

---

**文档生成时间**: 2026-01-14
**分析范围**: `/home/qy/src/Infini/InfiniCore/src/utils-test/`
**文档类型**: 叶子节点分析（Leaf Node Analysis）
**依赖文档**: `/home/qy/src/Infini/InfiniCore/src/README_ANALYSIS.md`
