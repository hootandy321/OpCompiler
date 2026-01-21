# 目录: infiniop-test 架构全景

## 1. 子系统职责

infiniop-test 是 InfiniOp 算子库的**自动化测试与基准测试框架**。该子系统的核心职责是：

1. **算子正确性验证**：通过 GGUF 格式的测试用例文件，批量验证各类算子在不同硬件后端（CPU、NVIDIA GPU、Cambricon、Ascend、Metax 等）上的计算正确性
2. **性能基准测试**：支持预热（warmup）和多次迭代运行，测量算子执行时间，提供性能基准数据
3. **跨硬件兼容性测试**：统一测试接口，验证算子在多硬件平台上的功能一致性
4. **测试用例标准化管理**：采用 GGUF 二进制格式封装测试数据（张量、属性、期望输出），实现测试用例的高效存储和加载

该子系统是 InfiniCore 生态系统中的质量保障组件，确保底层算子库在不同硬件平台上的功能正确性和性能稳定性。

## 2. 模块导航

### 2.1 核心模块

* **include/**（头文件接口层）
  * **功能**：定义测试框架的核心数据结构、接口和宏，供测试实现和主程序使用
  * **职责**：提供测试框架的抽象接口，包括测试用例基类、结果报告、GGUF 文件解析、张量管理等
  * **关键组件**：
    * `test.hpp`：测试框架核心接口（Test 基类、Result 类、测试注册宏）
    * `ops.hpp`：算子测试注册表（声明所有支持的算子测试）
    * `gguf.hpp`：GGUF 文件格式解析器（读取测试用例数据）
    * `tensor.hpp`：张量抽象层（封装设备内存、数据类型转换、跨设备数据传输）
    * `utils.hpp`：工具函数（类型转换、数值提取）
    * `file_mapping.hpp`：跨平台文件内存映射（零拷贝加载测试数据）

* **src/**（核心实现层）
  * **功能**：实现测试框架的核心逻辑和驱动程序
  * **职责**：提供测试执行引擎、结果报告、设备管理等实现
  * **关键组件**：
    * `main.cpp`：命令行入口（参数解析、设备初始化、测试调度）
    * `test.cpp`：测试执行引擎（加载 GGUF 测试用例、分发到具体算子测试、结果验证）
    * `tensor.cpp`：张量管理实现（内存分配、设备传输、数据访问）
    * `gguf.cpp`：GGUF 文件解析实现（读取元数据和张量数据）
    * `file_mapping.cpp`：文件内存映射实现（Linux mmap/Windows CreateFileMapping）

* **src/ops/**（算子测试实现层）
  * **功能**：实现各个算子的具体测试逻辑
  * **职责**：为每个算子提供测试用例构建、算子描述符创建、执行调用、结果验证
  * **文档状态**：该目录包含 17 个算子测试实现，无独立文档，通过源码分析获取功能信息
  * **支持的算子类型**：
    * **基础运算**：`add.cpp`、`sub.cpp`、`mul.cpp`、`zeros.cpp`、`ones.cpp`
    * **激活函数**：`sigmoid.cpp`、`silu.cpp`、`swiglu.cpp`
    * **矩阵运算**：`gemm.cpp`（通用矩阵乘法）
    * **归一化**：`rms_norm.cpp`、`causal_softmax.cpp`
    * **张量操作**：`clip.cpp`、`rearrange.cpp`
    * **位置编码**：`rope.cpp`（旋转位置编码）
    * **采样与路由**：`random_sample.cpp`、`topkrouter.cpp`、`topksoftmax.cpp`

## 3. 架构逻辑图解

### 3.1 整体工作流程

```
命令行入口 (main.cpp)
    ↓
[参数解析] → 解析 GGUF 文件路径、设备类型、性能测试参数
    ↓
[初始化 InfiniRT] → infinirtInit() 初始化运行时环境
    ↓
测试执行引擎 (test.cpp::runAllTests)
    ↓
[GGUF 文件加载] (gguf.cpp)
    → FileMapping 零拷贝映射文件到内存
    → 解析 GGUF 头部、元数据、张量信息
    → 构建 test_count 属性和测试用例索引
    ↓
测试循环 (对每个 test_id)
    ↓
[测试用例分发] (test.cpp::runTest)
    → 读取 op_name（如 "gemm"）
    → 从 TEST_BUILDERS 注册表查找对应测试构建器
    ↓
[测试构建] (各算子测试的 build 函数)
    → 从 GGUF 元数据提取算子属性（alpha、beta 等）
    → 从 GGUF 张量数据构建输入张量（Tensor 对象）
    → 创建测试实例（继承自 base::Test）
    ↓
[测试执行] (各算子测试的 run 函数)
    → 1. 数据准备：将输入张量传输到目标设备（Tensor::to()）
    → 2. 创建算子描述符：infiniopCreateXXXDescriptor()
    → 3. 预热（可选）：执行 warmups 次预热调用
    → 4. 性能测试：执行 iterations 次调用并计时
    → 5. 结果验证：将输出张量传回主机，与期望输出比较（allClose）
    ↓
结果报告 (Result::toString)
    → 输出测试状态（PASS/FAILED）
    → 输出执行时间
    → 输出错误信息（如有）
    ↓
汇总统计 → 输出通过/失败测试数量
```

### 3.2 核心组件交互关系

#### 3.2.1 测试注册机制

```
ops.hpp（声明层）
    → DECLARE_INFINIOP_TEST(gemm) 宏定义
        → 生成 gemm::Test 类接口（build、attribute_names、tensor_names、output_names）

src/ops/gemm.cpp（实现层）
    → 实现 gemm::Test::build()：从 GGUF 数据构建测试实例
    → 实现 gemm::Test::run()：执行 GEMM 测试逻辑

test.cpp（注册层）
    → TEST_BUILDERS 全局映射表
        → 键：算子名称（"gemm"）
        → 值：TestBuilder 结构体（build 函数、属性名列表、张量名列表、输出名列表）

ops.hpp 宏定义
    → TEST_BUILDER_MAPPINGS 宏
        → 展开为所有算子的 REGISTER_INFINIOP_TEST 调用
        → 初始化 TEST_BUILDERS 映射表
```

#### 3.2.2 GGUF 测试数据组织

```
GGUF 文件结构
├── 元数据（Key-Value 对）
│   ├── test_count：测试用例总数
│   └── test.<id>.*：每个测试的元数据
│       ├── op_name：算子名称（如 "gemm"）
│       ├── alpha：算子属性（浮点数）
│       ├── beta：算子属性（浮点数）
│       └── test.<id>.<tensor_name>.shape：张量形状
└── 张量数据（二进制块）
    └── test.<id>.<tensor_name>：张量原始数据
        ├── a：输入张量 A
        ├── b：输入张量 B
        ├── c：输入张量 C（可选）
        └── ans：期望输出张量
```

#### 3.2.3 设备管理与数据流转

```
设备初始化
    → infiniRT 初始化（infinirtInit）
    → 设置目标设备（infinirtSetDevice(device, device_id)）
    → 创建 InfiniOp 句柄（infiniopCreateHandle）

张量数据流（以 GEMM 为例）
    1. [文件] GGUF 文件通过 FileMapping 映射到内存（零拷贝）
    2. [解析] GGUFFileReader 解析张量元信息（形状、类型、偏移）
    3. [创建] Tensor 构造函数创建张量对象（指向文件映射内存）
    4. [传输] Tensor::to(device, device_id) 将数据从主机复制到设备
        → 分配设备内存（Memory 构造函数）
        → 调用 infiniRT 的数据传输接口
    5. [执行] InfiniOp 算子在设备上执行（操作设备内存）
    6. [验证] 将结果张量传回主机，与期望输出比较
```

#### 3.2.4 测试结果验证流程

```
算子执行完成
    ↓
[数据回传] output_tensor->to(INFINI_DEVICE_CPU)
    → 将设备内存复制回主机
    ↓
[结果比较] allClose(actual, expected, rtol, atol)
    → 遍历张量元素
    → 对每个元素计算：|actual - expected| ≤ atol + rtol × |expected|
    → 记录不匹配元素的索引和偏差
    ↓
[结果判定]
    → 全部通过：返回 TestStatus::PASS
    → 存在偏差：返回 TestStatus::RESULT_INCORRECT（附带错误信息）
```

### 3.3 扩展性设计

#### 3.3.1 添加新算子测试的步骤

1. **在 ops.hpp 中声明**：
   ```cpp
   DECLARE_INFINIOP_TEST(new_op)
   ```

2. **在 TEST_BUILDER_MAPPINGS 宏中注册**：
   ```cpp
   #define TEST_BUILDER_MAPPINGS \
       { \
           REGISTER_INFINIOP_TEST(gemm) \
           REGISTER_INFINIOP_TEST(new_op) \
       }
   ```

3. **在 src/ops/ 目录下实现 new_op.cpp**：
   ```cpp
   namespace infiniop_test::new_op {
       // 定义 Attributes 结构体（存储算子属性和张量）
       // 实现 Test::build()：从 GGUF 数据构建测试实例
       // 实现 Test::run()：执行测试逻辑
       // 实现 Test::attribute_names()：返回属性名列表
       // 实现 Test::tensor_names()：返回张量名列表
       // 实现 Test::output_names()：返回输出张量名列表
   }
   ```

4. **准备 GGUF 测试用例文件**：
   - 添加 `test.<id>.op_name = "new_op"` 元数据
   - 添加算子属性（如 `test.<id>.alpha`）
   - 添加输入张量数据（`test.<id>.input_a`、`test.<id>.input_b`）
   - 添加期望输出张量（`test.<id>.ans`）

#### 3.3.2 支持新硬件后端

测试框架已经通过 InfiniRT 抽象层支持多硬件，添加新硬件后端无需修改测试代码：

1. **InfiniRT 层**：实现新硬件的设备管理、内存分配、数据传输接口
2. **InfiniOp 层**：实现新硬件的算子内核
3. **测试框架**：通过命令行参数 `--<device>:id` 指定新硬件，自动调用对应后端

当前支持的硬件后端：CPU、NVIDIA、Cambricon、Ascend、Metax、Moore、Iluvatar、QY、Kunlun、Hygon。

### 3.4 错误处理机制

测试框架定义了详细的错误状态码：

- **PASS**：测试通过（结果在容差范围内）
- **TEST_INIT_FAILED**：测试用例初始化失败（缺少必需的属性或张量）
- **OP_CREATION_FAILED**：算子描述符创建失败（InfiniOp API 返回错误）
- **OP_EXECUTION_FAILED**：算子执行失败（运行时错误）
- **RESULT_INCORRECT**：结果不正确（超出容差范围）

每个测试用例独立执行，单个测试失败不会影响其他测试。最终报告统计通过/失败数量，并输出失败测试的详细错误信息。

### 3.5 性能测试模式

测试框架支持两种测试模式：

1. **正确性验证模式**（默认）：
   - `--warmup 0 --run 0`
   - 仅执行一次算子调用，验证结果正确性

2. **性能基准测试模式**：
   - `--warmup <n> --run <m>`
   - 先执行 n 次预热（warmup）调用，消除冷启动影响
   - 再执行 m 次正式调用，记录平均执行时间
   - 使用 std::chrono 高精度计时

性能测试结果在 `Result::toString()` 中输出，单位为微秒（us）。
