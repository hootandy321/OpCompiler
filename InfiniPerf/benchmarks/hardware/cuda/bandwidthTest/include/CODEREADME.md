# CUDA Helper Library Core Implementation Documentation

这是一个由 NVIDIA 官方提供的 CUDA 辅助工具库，专门用于 CUDA SDK 示例程序的初始化、错误检查、图像处理、计时和字符串解析等通用功能。该库通过跨平台抽象层（Windows/Linux/macOS）提供了统一的接口，简化了 CUDA 应用程序的开发流程。

## 1. Module Structure

- **`exception.h`**: 基于模板的异常处理包装器，提供带位置信息的异常抛出机制
- **`helper_cuda.h`**: CUDA 运行时和驱动 API 的核心辅助函数，包括设备初始化、错误处理、架构查询
- **`helper_functions.h`**: 通用工具函数的聚合头文件，统一包含其他辅助模块
- **`helper_image.h`**: PGM/PPM 图像文件的加载、保存、比较和二进制数据操作
- **`helper_string.h`**: 命令行参数解析、文件路径查找和字符串处理工具
- **`helper_timer.h`**: 跨平台的高精度计时器实现（Windows QueryPerformanceCounter/Linux gettimeofday）

## 2. Core Classes

### `Exception<Std_Exception>`
- **Location**: `exception.h`
- **Primary Function**: 提供带文件名和行号信息的类型安全异常抛出包装器，继承自标准库异常类
- **Key Members**:
  - 无公共成员变量（仅继承自 Std_Exception）
- **Core Methods**:
  - `static void throw_it(const char *file, const int line, const char *detailed)`: 抛出异常并附加位置信息，使用 std::stringstream 构建详细错误消息（包含文件路径、行号、详细描述），复杂度 O(1)
  - `static void throw_it(const char *file, const int line, const std::string &detailed)`: 字符串重载版本，内部调用 const char* 版本
  - `virtual ~Exception() throw()`: 虚析构函数，确保正确的异常对象清理
- **Lifecycle**: 私有构造函数，只能通过静态方法 `throw_it()` 创建实例；遵循 RAII 原则，异常对象在 catch 块结束后自动销毁

### `StopWatchInterface`
- **Location**: `helper_timer.h`
- **Primary Function**: 计时器的抽象基类，定义跨平台计时器的统一接口
- **Key Members**:
  - 无成员变量（纯接口类）
- **Core Methods**:
  - `virtual void start() = 0`: 启动计时，记录开始时间点
  - `virtual void stop() = 0`: 停止计时，累计时间差到总计时间
  - `virtual void reset() = 0`: 重置所有计数器为零，如果计时器正在运行则重新捕获当前时间点
  - `virtual float getTime() = 0`: 返回自上次 reset() 或创建以来的总时间（毫秒），如果正在运行则包含当前会话的已过时间
  - `virtual float getAverageTime() = 0`: 返回已完成会话的平均时间（总时间/会话数），复杂度 O(1)
- **Lifecycle**: 纯虚接口，通过 `sdkCreateTimer()` 工厂函数创建具体平台实例（StopWatchWin 或 StopWatchLinux）

### `StopWatchWin`
- **Location**: `helper_timer.h`
- **Primary Function**: Windows 平台的高精度计时器实现，基于 QueryPerformanceCounter API
- **Key Members**:
  - `LARGE_INTEGER start_time`: 计时开始的性能计数器值
  - `LARGE_INTEGER end_time`: 计时结束的性能计数器值
  - `float diff_time`: 最后一次 start-stop 会话的时间差（毫秒）
  - `float total_time`: 自上次 reset() 以来所有会话的累计时间
  - `bool running`: 计时器运行状态标志
  - `int clock_sessions`: 已完成的 start-stop 会话计数
  - `double freq`: 性能计数器频率（转换为毫秒单位）
  - `static bool freq_set`: 全局频率查询标志（所有实例共享）
- **Core Methods**:
  - `StopWatchWin()`: 构造函数，首次调用时通过 `QueryPerformanceFrequency()` 获取系统频率并缓存，频率值转换为 ticks/ms 单位
  - `void start()`: 调用 `QueryPerformanceCounter()` 记录 start_time，设置 running=true
  - `void stop()`: 调用 `QueryPerformanceCounter()` 记录 end_time，计算时间差（(end.QuadPart - start.QuadPart) / freq），累加到 total_time，clock_sessions++，running=false
  - `void reset()`: 重置 diff_time=0, total_time=0, clock_sessions=0，如果正在运行则重新捕获 start_time
  - `float getTime()`: 返回 total_time，如果 running=true 则加上当前会话的已过时间（temp.QuadPart - start_time.QuadPart）/ freq
  - `float getAverageTime()`: 返回 total_time / clock_sessions（如果 clock_sessions>0），否则返回 0.0f
- **Lifecycle**: 通过 `sdkCreateTimer()` 动态分配，必须通过 `sdkDeleteTimer()` 释放；频率在第一个实例创建时全局查询一次

### `StopWatchLinux`
- **Location**: `helper_timer.h`
- **Primary Function**: Linux/macOS 平台的计时器实现，基于 gettimeofday() 系统调用
- **Key Members**:
  - `struct timeval start_time`: 计时开始的时间戳（秒 + 微秒）
  - `float diff_time`: 最后一次会话的时间差
  - `float total_time`: 累计总时间
  - `bool running`: 运行状态标志
  - `int clock_sessions`: 会话计数
- **Core Methods**:
  - `StopWatchLinux()`: 默认构造函数，初始化所有成员为默认值
  - `void start()`: 调用 `gettimeofday(&start_time, 0)`，设置 running=true
  - `void stop()`: 调用 `getDiffTime()` 计算时间差，累加到 total_time，clock_sessions++，running=false
  - `void reset()`: 重置所有计数器，如果运行中则重新 gettimeofday()
  - `float getTime()`: 返回 total_time +（当前运行时间，如果 running）
  - `float getAverageTime()`: 返回 total_time / clock_sessions
  - `float getDiffTime()`: 私有辅助方法，计算当前时间与 start_time 的差值（1000.0 * (t.tv_sec - start_time.tv_sec) + 0.001 * (t.tv_usec - start_time.tv_usec)）
- **Lifecycle**: 通过 `sdkCreateTimer()` 创建，使用 gettimeofday() 提供微秒级精度

### `ConverterFromUByte<T>`
- **Location**: `helper_image.h` (namespace helper_image_internal)
- **Primary Function**: 模板特化结构体，将 unsigned char 数据转换为指定类型
- **Key Members**:
  - 无成员变量
- **Core Methods**:
  - `float operator()(const unsigned char &val)`: 函数调用运算符
    - `T=unsigned char` 特化：直接返回值（无转换）
    - `T=float` 特化：返回 `static_cast<float>(val) / 255.0f`（归一化到 [0, 1]）
- **Lifecycle**: 无状态函数对象，由 `std::transform()` 算法临时构造

### `ConverterToUByte<T>`
- **Location**: `helper_image.h` (namespace helper_image_internal)
- **Primary Function**: 将指定类型转换回 unsigned char
- **Key Members**:
  - 无成员变量
- **Core Methods**:
  - `unsigned char operator()(const T &val)`: 转换运算符
    - `T=unsigned char` 特化：直接返回（恒等转换）
    - `T=float` 特化：返回 `static_cast<unsigned char>(val * 255.0f)`（反归一化）
- **Lifecycle**: 无状态函数对象，用于图像数据保存时的格式转换

## 3. API Interface

```cpp
// CUDA Error Handling (helper_cuda.h)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line);
// 检查 CUDA API 调用返回值，非零时打印错误信息并调用 exit(EXIT_FAILURE)
// 参数: result - CUDA API 返回的错误码, func - 函数名（通过宏自动注入）, file - 源文件名, line - 行号

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// 宏包装器，自动捕获函数名、文件名和行号

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
// 检查 cudaGetLastError()，捕获最近的 CUDA 错误并打印详细信息

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line);
// 内联函数，调用 cudaGetLastError() 并检查是否为 cudaSuccess

#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)
// 非退出版本的 getLastCudaError，仅打印错误但不终止程序

// CUDA Device Management (helper_cuda.h)
inline int gpuDeviceInit(int devID);
// 初始化指定 CUDA 设备，检查设备有效性、计算模式、计算能力
// 返回: 设备 ID（成功）或负值（失败）

inline int gpuGetMaxGflopsDeviceId();
// 遍历所有 CUDA 设备，计算性能指标（multiProcessorCount * coresPerSM * clockRate）
// 返回: 性能最高的设备 ID

inline int findCudaDevice(int argc, const char **argv);
// 高级设备选择函数：如果命令行指定 -device=N 则使用该设备，否则自动选择最高性能设备
// 返回: 选中的设备 ID

inline int findIntegratedGPU();
// 查找集成 GPU（用于 CUDA-OpenGL 互操作）
// 返回: 集成 GPU 的设备 ID，未找到返回 -1

inline bool checkCudaCapabilities(int major_version, int minor_version);
// 检查当前设备是否满足最低计算能力要求
// 返回: true 如果当前设备能力 >= 要求

// Architecture Query Functions (helper_cuda.h)
inline int _ConvertSMVer2Cores(int major, int minor);
// 根据 SM 版本返回每个 SM 的 CUDA 核心数
// 映射表: Kepler(192), Maxwell(128), Pascal(64/128), Volta(64), Turing(64), Ampere(64/128), Hopper(128), Blackwell(128)
// 算法: 线性搜索静态数组 nGpuArchCoresPerSM[]，复杂度 O(n)，n 为架构数量（~21）

inline const char* _ConvertSMVer2ArchName(int major, int minor);
// 返回架构名称字符串（"Kepler", "Maxwell", "Volta", "Ampere", "Hopper", "Blackwell" 等）
// 算法: 线性搜索静态数组 nGpuArchNameSM[]

// String Parsing Functions (helper_string.h)
inline int stringRemoveDelimiter(char delimiter, const char *string);
// 移除字符串开头的指定分隔符（如 '-'），返回第一个非分隔符字符的索引

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref);
// 检查命令行是否包含指定标志（支持 -flag 或 --flag，忽略大小写）
// 算法: 遍历 argv[1..argc-1]，使用 STRNCASECMP 比较标志名，支持带等号的参数（如 -device=0）

template <class T>
inline bool getCmdLineArgumentValue(const int argc, const char **argv, const char *string_ref, T *value);
// 提取命令行标志的数值参数（自动处理等号和偏移）
// 例子: -device=0 => 提取 0 到 value

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);
// 专用版本，返回 int 值（未找到返回 0）

inline float getCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref);
// 专用版本，返回 float 值

inline bool getCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval);
// 提取字符串参数值

inline char *sdkFindFilePath(const char *filename, const char *executable_path);
// 在预定义的搜索路径列表中查找文件（支持相对路径和可执行文件名替换）
// 搜索路径: ./, ./data/, ../../Samples/<executable_name>/, ../../Common/data/ 等
// 算法: 遍历约 60 个预定义路径，替换 <executable_name> 宏，尝试打开文件（fopen "rb" 模式）
// 返回: 动态分配的完整路径字符串（调用者需 free()），未找到返回 NULL

// Image Processing Functions (helper_image.h)
template <class T>
inline bool sdkLoadPGM(const char *file, T **data, unsigned int *w, unsigned int *h);
// 加载 PGM（灰度）图像文件，自动转换数据类型
// 流程: __loadPPM() 读取原始数据 -> malloc() 分配目标缓冲区 -> std::transform() + ConverterFromUByte 转换
// 支持 T=unsigned char（直接复制）或 T=float（归一化）

template <class T>
inline bool sdkLoadPPM4(const char *file, T **data, unsigned int *w, unsigned int *h);
// 加载 PPM（彩色）图像并填充第 4 通道（Alpha=0）
// 流程: 读取 RGB 数据 -> 分配 RGBA 缓冲区（w*h*4） -> 逐像素填充，第 4 字节设为 0

template <class T>
inline bool sdkSavePGM(const char *file, T *data, unsigned int w, unsigned int h);
// 保存灰度图像为 PGM 格式
// 流程: std::transform() + ConverterToUByte 转换 -> __savePPM() 写入文件头（"P5\n"）和数据

inline bool sdkSavePPM4ub(const char *file, unsigned char *data, unsigned int w, unsigned int h);
// 保存 RGBA 图像为 PPM（去除 Alpha 通道）
// 算法: 分配临时缓冲区（w*h*3），逐像素跳过第 4 字节复制

// File I/O Functions (helper_image.h)
template <class T>
inline bool sdkReadFile(const char *filename, T **data, unsigned int *len, bool verbose);
// 读取文本文件中的浮点数序列
// 算法: fscanf(fh, "%f", &token) 循环读取到 std::vector，弹出最后一个重复元素，malloc() 分配目标缓冲区，memcpy() 复制
// 复杂度: O(n)，n 为文件中的浮点数个数

template <class T>
inline bool sdkReadFileBlocks(const char *filename, T **data, unsigned int *len, unsigned int block_num, unsigned int block_size, bool verbose);
// 二进制分块读取，使用 fseek() 定位到 block_num * block_size 偏移量
// 应用场景: 大文件的分块加载

template <class T, class S>
inline bool sdkWriteFile(const char *filename, const T *data, unsigned int len, const S epsilon, bool verbose, bool append = false);
// 写入文本文件，首行写入 epsilon 值作为注释，后续逐元素写入

inline void sdkDumpBin(void *data, unsigned int bytes, const char *filename);
// 二进制转储函数，fwrite() 写入原始字节

// Data Comparison Functions (helper_image.h)
template <class T, class S>
inline bool compareData(const T *reference, const T *data, const unsigned int len, const S epsilon, const float threshold);
// 比较两个数组是否在 epsilon 容差内相等
// 算法: 逐元素计算差值 reference[i] - data[i]，检查是否在 [-epsilon, +epsilon] 范围内
// 如果 threshold > 0，允许一定比例的错误（error_count <= len * threshold）

template <class T, class S>
inline bool compareDataAsFloatThreshold(const T *reference, const T *data, const unsigned int len, const S epsilon, const float threshold);
// 浮点数专用比较版本，使用 fabs(diff) < max(epsilon, __MIN_EPSILON_ERROR)
// 最小容差 __MIN_EPSILON_ERROR = 1e-3f，避免 epsilon=0 时的过严格比较

inline bool sdkCompareL2fe(const float *reference, const float *data, const unsigned int len, const float epsilon);
// L2 范数（欧几里得范数）相对误差比较
// 算法: 计算 sqrt(Σ(ref[i] - data[i])²) / sqrt(Σref[i]²)，检查是否 < epsilon
// 复杂度: O(n)，两次遍历（计算误差和参考范数）

inline bool sdkCompareBin2BinUint(const char *src_file, const char *ref_file, unsigned int nelements, const float epsilon, const float threshold, char *exec_path);
// 二进制文件比较（unsigned int 类型），使用 sdkFindFilePath() 定位参考文件

inline bool sdkCompareBin2BinFloat(const char *src_file, const char *ref_file, unsigned int nelements, const float epsilon, const float threshold, char *exec_path);
// 二进制文件比较（float 类型），使用 compareDataAsFloatThreshold()

inline bool sdkComparePPM(const char *src_file, const char *ref_file, const float epsilon, const float threshold, bool verboseErrors);
// PPM 图像文件比较，自动加载 4 通道图像并调用 compareData()

inline bool sdkComparePGM(const char *src_file, const char *ref_file, const float epsilon, const float threshold, bool verboseErrors);
// PGM 灰度图像比较

// Timer Management (helper_timer.h)
inline bool sdkCreateTimer(StopWatchInterface **timer_interface);
// 工厂函数，根据平台创建 StopWatchWin 或 StopWatchLinux 实例
// 返回: 成功返回 true，timer_interface 指向新创建的对象

inline bool sdkDeleteTimer(StopWatchInterface **timer_interface);
// 销毁计时器并置空指针

inline bool sdkStartTimer(StopWatchInterface **timer_interface);
// 启动计时器

inline bool sdkStopTimer(StopWatchInterface **timer_interface);
// 停止计时器

inline bool sdkResetTimer(StopWatchInterface **timer_interface);
// 重置计时器

inline float sdkGetTimerValue(StopWatchInterface **timer_interface);
// 获取累计时间（毫秒）

inline float sdkGetAverageTimerValue(StopWatchInterface **timer_interface);
// 获取平均时间（毫秒）

// Error Enum Conversion (helper_cuda.h)
static const char *_cudaGetErrorEnum(cudaError_t error); // CUDA Runtime API
static const char *_cudaGetErrorEnum(CUresult error);    // CUDA Driver API
static const char *_cudaGetErrorEnum(cublasStatus_t error); // cuBLAS
static const char *_cudaGetErrorEnum(cufftResult error);     // cuFFT
static const char *_cudaGetErrorEnum(cusparseStatus_t error); // cuSPARSE
static const char *_cudaGetErrorEnum(cusolverStatus_t error); // cuSOLVER
static const char *_cudaGetErrorEnum(curandStatus_t error);   // cuRAND
static const char *_cudaGetErrorEnum(nvjpegStatus_t error);   // nvJPEG
static const char *_cudaGetErrorEnum(NppStatus error);        // NPP
// 函数重载集合，将各 CUDA 库的错误码转换为可读字符串
// 实现方式: switch-case 映射，返回常量字符串字面量

// Utility Macros (helper_cuda.h)
#define MAX(a, b) (a > b ? a : b)
// 最大值宏

inline int ftoi(float value);
// 浮点转整数，四舍五入（正数 +0.5，负数 -0.5）

// Exit Code Definition (多个头文件)
#define EXIT_WAIVED 2
// 测试跳过退出码（用于设备不支持等场景）
```

## 4. Usage Example

```cpp
// Example: Using CUDA Helper Library for Bandwidth Testing
#include <helper_cuda.h>    // CUDA device management and error checking
#include <helper_timer.h>   // High-precision timer
#include <helper_string.h>  // Command line parsing
#include <helper_functions.h>

int main(int argc, const char **argv) {
    // 1. Find and initialize CUDA device
    // Automatically selects highest-performance GPU unless -device=N specified
    int devID = findCudaDevice(argc, argv);
    printf("Using CUDA Device %d\n", devID);

    // 2. Parse command line arguments
    // Supports formats: -device=0, --mode=fast, -iterations=100
    int iterations = 100;
    if (checkCmdLineFlag(argc, argv, "iterations")) {
        iterations = getCmdLineArgumentInt(argc, argv, "iterations=");
    }

    bool useDMATransfer = checkCmdLineFlag(argc, argv, "dma");

    // 3. Allocate and initialize memory with error checking
    float *d_data;
    size_t bytes = 1024 * 1024 * sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&d_data, bytes));

    // Initialize data on host
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < 1024 * 1024; i++) {
        h_data[i] = i * 1.0f;
    }

    // 4. Create high-precision timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // 5. Benchmark loop
    sdkStartTimer(&timer);
    for (int i = 0; i < iterations; i++) {
        checkCudaErrors(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }
    sdkStopTimer(&timer);

    // 6. Get timing results
    float totalTime = sdkGetTimerValue(&timer);      // Total time across all iterations
    float avgTime = sdkGetAverageTimerValue(&timer); // Average time per iteration

    printf("Transfer time: %.3f ms (avg: %.3f ms)\n", totalTime, avgTime);

    // Calculate bandwidth in GB/s
    float bandwidth = (bytes * iterations) / (avgTime / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    // 7. Cleanup
    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaFree(d_data));
    free(h_data);

    // Check for any CUDA errors that occurred during execution
    getLastCudaError("CUDA kernel launch failed");

    return EXIT_SUCCESS;
}

// Example: Loading and Comparing Images
#include <helper_image.h>

void imageProcessingExample() {
    unsigned char *srcImage = NULL, *refImage = NULL;
    unsigned int width, height;

    // Load 4-channel RGBA images
    const char *srcFile = "output.ppm";
    const char *refFile = "reference.ppm";

    if (sdkLoadPPM4ub(srcFile, &srcImage, &width, &height)) {
        printf("Loaded source image: %dx%d\n", width, height);

        // Compare with reference
        const float epsilon = 1.0f;       // Allow 1-level difference
        const float threshold = 0.01f;    // Allow 1% pixel errors

        bool match = sdkComparePPM(srcFile, refFile, epsilon, threshold, true);

        if (match) {
            printf("Images match within tolerance!\n");
        } else {
            printf("Images differ significantly.\n");
        }

        free(srcImage);
    }

    // Save processed grayscale image
    unsigned char *grayData = /* ... processed data ... */;
    sdkSavePGM("output_gray.pgm", grayData, width, height);
}

// Example: File Path Discovery
void filePathExample() {
    // Automatically search for data files in standard locations
    char *dataPath = sdkFindFilePath("input_data.txt", argv[0]);

    if (dataPath) {
        printf("Found data file at: %s\n", dataPath);

        // Read data
        float *data;
        unsigned int len;
        if (sdkReadFile(dataPath, &data, &len, true)) {
            printf("Read %u float values\n", len);
            free(data);
        }

        free(dataPath);
    } else {
        printf("Data file not found in search paths\n");
    }
}

// Example: Custom Error Handling
void customErrorHandling() {
    // Use custom exception types with file/line information
    try {
        if (someCondition) {
            RUNTIME_EXCEPTION("File not found or permission denied");
        }

        if (logicalError) {
            LOGIC_EXCEPTION("Invalid state: device not initialized");
        }

        if (indexOutOfRange) {
            RANGE_EXCEPTION("Array index 100 exceeds size 50");
        }
    }
    catch (const std::exception& e) {
        // Exception message automatically includes:
        // - File path
        // - Line number
        // - Detailed description
        std::cerr << "Caught exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}
```

## 5. Implementation Details

- **Memory Management**:
  - 图像数据加载（`sdkLoadPGM/sdkLoadPPM4`）使用 `malloc()` 动态分配，调用者负责 `free()`，避免 C++ new/delete 混用
  - `sdkFindFilePath()` 返回 `malloc()` 分配的路径字符串，必须由调用者释放
  - 文件读取使用 `std::vector<T>` 作为中间存储，然后 `memcpy()` 到目标缓冲区，减少内存碎片
  - 计时器对象通过 `sdkCreateTimer()` 使用 `new` 分配，必须通过 `sdkDeleteTimer()` 释放

- **Concurrency**:
  - 无线程同步机制，所有函数假设单线程调用
  - CUDA API 调用本身是线程安全的，但 helper 函数不保护共享状态（如静态 freq_set）
  - 计时器实例可独立使用，但同一计时器对象的 start/stop 必须串行调用

- **Performance**:
  - **计时精度**: Windows 使用 `QueryPerformanceCounter()`（亚微秒级，依赖硬件频率），Linux 使用 `gettimeofday()`（微秒级）
  - **架构查询**: `_ConvertSMVer2Cores()` 使用线性搜索静态数组（21 个条目），复杂度 O(21) ≈ O(1)，仅在初始化时调用
  - **设备选择**: `gpuGetMaxGflopsDeviceId()` 遍历所有设备，复杂度 O(num_devices)，每个设备查询 5 个属性（computeMode, major, minor, multiProcessorCount, clockRate）
  - **图像转换**: 使用 `std::transform()` + 函数对象，编译器可内联优化，避免虚函数开销
  - **文件搜索**: `sdkFindFilePath()` 遍历约 60 个路径，每个路径尝试 `fopen()`，未找到文件时开销较大（应在初始化时缓存结果）

- **Error Handling**:
  - CUDA 错误使用模板函数 `check<T>()` 统一处理，非零错误码时打印详细错误信息（错误码、枚举名、函数签名、文件、行号）并 `exit(EXIT_FAILURE)`
  - 宏 `checkCudaErrors()` 自动注入 `__FILE__` 和 `__LINE__`，提供精确的错误位置
  - `getLastCudaError()` 检查异步错误的同步点，捕获 kernel launch 错误
  - `printLastCudaError()` 仅打印不退出，用于可选警告
  - 图像加载失败返回 `false`，调用者需检查返回值
  - 数据比较函数返回 `bool`，支持基于阈值的宽松比较（如允许 1% 像素错误）

- **Dependencies**:
  - **外部依赖**: CUDA Runtime API (`cuda_runtime.h`), CUDA Driver API (可选), Standard Template Library (`<algorithm>`, `<vector>`, `<fstream>`, `<sstream>`)
  - **内部依赖**: `helper_functions.h` 聚合包含其他所有 helper 头文件，`exception.h` 被 `helper_image.h` 和 `helper_timer.h` 依赖
  - **平台特定**:
    - Windows: `windows.h` (QueryPerformanceCounter/QueryPerformanceFrequency), `strcpy_s`, `fopen_s`, `sscanf_s`, `sprintf_s`
    - Linux/macOS: `sys/time.h` (gettimeofday), `string.h` (strcasecmp), 标准 POSIX 函数

- **Design Patterns**:
  - **模板方法模式**: `Exception<Std_Exception>` 使用模板参数包装不同异常类型（`std::runtime_error`, `std::logic_error`, `std::range_error`）
  - **策略模式**: `StopWatchInterface` 定义计时接口，`StopWatchWin` 和 `StopWatchLinux` 提供平台特定实现
  - **工厂模式**: `sdkCreateTimer()` 根据编译时宏选择具体计时器类
  - **函数对象模式**: `ConverterFromUByte<T>` 和 `ConverterToUByte<T>` 作为 `std::transform()` 的策略参数
  - **RAII**: 异常对象遵循 RAII，析构函数自动清理；计时器对象封装资源管理
  - **宏包装器**: `checkCudaErrors()`, `getLastCudaError()` 等宏简化 API 并自动捕获上下文信息
  - **零开销抽象**: 转换函数对象内联后无额外开销，编译器优化后等价于手写循环

- **跨平台兼容性**:
  - 使用宏抽象平台差异（`WIN32`, `_WIN32`, `WIN64`, `_WIN64`）
  - 字符串函数宏定义（`STRCASECMP`, `STRNCASECMP`, `STRCPY`, `FOPEN`, `SSCANF`, `SPRINTF`）统一接口
  - 文件路径分隔符自动处理（Windows 用 `\`，Unix 用 `/`）
  - 可执行文件名提取（Windows 去除 `.exe` 后缀，Unix 直接使用）
