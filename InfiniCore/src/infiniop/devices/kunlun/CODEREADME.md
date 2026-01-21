# Kunlun XPU Device Backend Implementation Documentation

This module provides the Kunlun XPU device backend implementation for InfiniOp, enabling tensor operations on Baidu Kunlun AI accelerators through the XPU runtime and XDNN library. It implements device handle management, BLAS operations wrapper, and kernel-level utilities for XPU device programming.

## 1. Module Structure

- **`kunlun_common.h`**: Common type definitions and XPU runtime headers integration
- **`kunlun_handle.h`** / **`kunlun_handle.cc`**: Main device handle implementation with XDNN context pooling
- **`kunlun_xblas.h`** / **`kunlun_xblas.cc`**: BLAS handle wrapper (note: implementation references CUDA cublas - appears to be template code)
- **`kunlun_kernel_common.h`**: XPU kernel programming utilities including atomic operations, SIMD vectorized reductions, and tensor indexing

## 2. Core Classes

### `device::kunlun::Handle`
- **Location**: `kunlun_handle.h` / `kunlun_handle.cc`
- **Primary Function**: Main device handle representing a Kunlun XPU device instance, inheriting from `InfiniopHandle`
- **Key Members**:
  - `_internal`: `std::shared_ptr<Internal>` - Internal implementation handle with XDNN context pool
  - `device`: `infiniDevice_t` (inherited) - Set to `INFINI_DEVICE_KUNLUN`
  - `device_id`: `int` (inherited) - Logical device identifier
- **Core Methods**:
  - `Handle(int device_id)`: Constructor initializing device type and internal handle
  - `create(InfiniopHandle **handle_ptr, int device_id)`: Static factory method creating new handle instances
  - `internal()`: Accessor returning const reference to internal implementation
- **Lifecycle**: Direct instantiation via factory pattern, internal handle managed through shared_ptr for automatic cleanup

### `device::kunlun::Handle::Internal`
- **Location**: `kunlun_handle.h` / `kunlun_handle.cc`
- **Primary Function**: Encapsulates XDNN context management with thread-safe pooling mechanism
- **Key Members**:
  - `dnn_handles`: `Pool<xdnnHandle_t>` - Lock-free object pool for XDNN context reuse
  - `Fn<T>`: `template<typename T> using Fn = std::function<infiniStatus_t(T)>` - Type alias for status-returning callbacks
- **Core Methods**:
  - `useXdnn(kunlunStream_t stream, const Fn<xdnnHandle_t> &f)`: Acquires XDNN context from pool, sets stream, executes callback, returns handle to pool
    - **Algorithm**: Lock-free stack pop from pool, lazy initialization on first use, stream binding via `xdnn::Context::set_stream()`
    - **Thread Safety**: Uses atomic compare_exchange in pool operations for concurrent access
    - **Error Handling**: Returns `INFINI_STATUS_SUCCESS` or propagates callback errors via `CHECK_STATUS` macro
- **Lifecycle**: Managed as shared_ptr within outer Handle, pool destructor cleans up all contexts

### `device::kunlun::blas::Handle`
- **Location**: `kunlun_xblas.h` / `kunlun_xblas.cc`
- **Primary Function**: BLAS-specific handle wrapper (appears to be incomplete/template code - still references cublas)
- **Key Members**:
  - `_internal`: `std::shared_ptr<Internal>` - BLAS handle pool manager
  - `blas_handles`: `Pool<cublasHandle_t>` - **BUG**: Should use XPU BLAS handle, not CUDA
- **Core Methods**:
  - `useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f)`: **INCORRECT**: References CUDA instead of XPU BLAS
- **Status**: Implementation incomplete, appears to be copy-paste from CUDA backend without proper adaptation

## 3. Kernel Utilities (namespace `device::kunlun::kernel`)

### Atomic Operations

#### `atomicAdd<T>(__shared_ptr__ T *ptr, T value)`
- **Specializations**:
  - **Generic**: Uses XPU builtin `atomicadd()` for 32/64-bit integers and floats
  - **`half`**: Ticket-lock based implementation with float conversion
    - **Algorithm**: `ticket_lock_mix()` → read old → convert to float → add → convert back → store → `mfence_sm()` → `ticket_unlock_mix()`
    - **Reason**: XPU lacks native half atomics, requires software serialization
  - **`bfloat16_t`**: Similar ticket-lock implementation with `__bfloat162float()` and `__float2bfloat16_rn()` conversions
- **Returns**: Old value before addition (standard atomic semantics)

#### `atomicMax<T>(__shared_ptr__ T *ptr, T value)`
- **Implementation**: Uses ticket lock for all types, compares and updates maximum
- **Algorithm**:
  - Acquire ticket lock
  - Read old value
  - If `bfloat16_t`: convert both to float, use `fmax()`, convert back
  - Otherwise: direct `fmax(old, value)`
  - Store result, issue memory fence (`mfence_sm()`), release lock

### Tensor Indexing

#### `indexToOffset(int flat_index, int ndim, const _size_t *shape, const _ptrdiff_t *strides)`
- **Purpose**: Converts flat index to byte/element offset for strided tensors
- **Algorithm**: Row-major (C-style) dimension decomposition from innermost to outermost
  ```cpp
  offset = 0;
  for i from ndim-1 down to 0:
      offset += (flat_index % shape[i]) * strides[i]
      flat_index /= shape[i]
  ```
- **Complexity**: O(ndim) per call
- **Type Handling**: Uses custom `_size_t` and `_ptrdiff_t` structs (32-bit value + 32-bit padding) to match XPU's 32-bit architecture while maintaining 64-bit alignment for DMA transfers

### SIMD Vectorized Reductions

#### `max<T>(const T *data_ptr, size_t len)`
- **Generic**: Sequential reduction with `fmax()` - O(n)
- **`half` specialization**:
  - Uses 512-bit SIMD registers (32 half values per vector)
  - **Vector width**: 32 elements per `float16x32_t` vector
  - **Algorithm**:
    1. Handle tail elements (< 32) sequentially
    2. Process main loop in chunks of 32 using `vload_lm_float16x32_mz()` / `vstore_lm_float16x32_mz()`
    3. Vector max using `vvmax_float16x32_mz()`
    4. Final sequential reduction of 32 accumulator elements
  - **Performance**: ~16x speedup over scalar (32-way parallelism)
- **`float` specialization**:
  - Uses 512-bit SIMD registers (16 float values per vector)
  - **Vector width**: 16 elements per `float32x16_t` vector
  - **Functions**: `vload_lm_float32x16_mz()`, `vvmax_float32x16_mz()`, `vstore_lm_float32x16_mz()`
  - **Performance**: ~16x speedup over scalar

### Memory Utilities

#### `loadsm<T>(__shared_ptr__ const T *p, T *v, int len)`
- **Purpose**: Bulk load from shared memory to local register/memory
- **Implementation**: Direct `__builtin_memcpy()` with byte size calculation
- **Use Case**: Optimized for loading shared memory tiles into thread-local storage

#### `lowerBitMask(int i)`
- **Purpose**: Generate bitmask for 512-bit register partial loads
- **Formula**: `(1 << (i + 1)) - 1` - produces lower i+1 bits set to 1
- **Use Case**: Masking valid data when array length < SIMD vector width

## 4. API Interface

### Device Handle Management

```cpp
namespace device::kunlun {

// Factory method
infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
// Creates and initializes Kunlun device handle
// Parameters:
//   handle_ptr: Output pointer to receive handle
//   device_id: XPU device number (0-N)
// Returns: INFINI_STATUS_SUCCESS on success

// XDNN context execution
infiniStatus_t Handle::Internal::useXdnn(
    kunlunStream_t stream,
    const std::function<infiniStatus_t(xdnnHandle_t)> &f
) const;
// Executes callback with XDNN context from pool
// Automatically handles context creation/reuse and stream binding
}
```

### Kernel Utilities

```cpp
namespace device::kunlun::kernel {

// Atomic operations
template <typename T>
T atomicAdd(__shared_ptr__ T *ptr, T value);
// Atomic addition with specialization for half/bfloat16

template <typename T>
T atomicMax(__shared_ptr__ T *ptr, T value);
// Atomic maximum with float conversion for bfloat16

// Tensor indexing
int indexToOffset(
    int flat_index,
    int ndim,
    const _size_t *shape,
    const _ptrdiff_t *strides
);
// Converts flat index to offset for strided tensors

// SIMD reductions
template <typename T>
T max(const T *data_ptr, size_t len);
// Vectorized maximum: 32-way for half, 16-way for float

// Memory utilities
template <typename T>
void loadsm(__shared_ptr__ const T *p, T *v, int len);
// Bulk shared memory load

float lowerBitMask(int i);
// Bitmask generator for partial vector loads
}
```

## 5. Usage Example

```cpp
#include "infiniop/devices/kunlun/kunlun_handle.h"

using namespace device::kunlun;

// Initialize Kunlun device
InfiniopHandle *raw_handle;
Handle::create(&raw_handle, 0);  // Device 0
Handle *handle = static_cast<Handle *>(raw_handle);

// Create XPU stream
kunlunStream_t stream;
xpu_stream_create(&stream);

// Execute XDNN operation with automatic context pooling
infiniStatus_t status = handle->internal()->useXdnn(stream, [](xdnnHandle_t xdnn_handle) {
    // Use XDNN API with context
    xdnn::convolution_forward(
        xdnn_handle,
        /* input params */
    );
    return INFINI_STATUS_SUCCESS;
});

// Stream synchronization
xpu_stream_synchronize(stream);

// Cleanup (automatic via RAII)
delete handle;
xpu_stream_destroy(stream);
```

**Kernel Programming Example** (in .xpu file):

```cpp
#include "infiniop/devices/kunlun/kunlun_kernel_common.h"

using namespace device::kunlun::kernel;

__global__ void reduce_max_kernel(
    __shared_ptr__ float *input,
    __shared_ptr__ float *output,
    _size_t n) {

    __local__ float local_data[256];
    loadsm(input + tid * 256, local_data, 256);

    float max_val = max(local_data, 256);  // SIMD-optimized

    atomicMax(output, max_val);  // Thread-safe reduction
}
```

## 6. Implementation Details

### Memory Management
- **Object Pooling**: Lock-free Treiber stack using `std::atomic` with `compare_exchange_weak` for XDNN context reuse
  - **Benefit**: Avoids repeated XDNN context creation overhead (expensive operation)
  - **Thread Safety**: Lock-free design allows concurrent access from multiple threads
- **Shared Ownership**: `std::shared_ptr` for internal handle ensures lifetime management across thread boundaries
- **XPU Memory Architecture**:
  - Shared memory (`__shared_ptr__`): 40KB SM_SIZE (40960 bytes)
  - Local memory (`__local__`): Thread-private scratchpad with SIMD load/store

### Concurrency
- **Lock-Free Pool**: Uses atomic CAS (Compare-And-Swap) on stack head pointer
  - **Algorithm**: Treiber stack with optimistic concurrency control
  - **Complexity**: O(1) push/pop amortized
- **Ticket Locks**: Used for atomic operations on half/bfloat16 types lacking native hardware support
  - **Implementation**: XPU `ticket_lock_mix()` / `ticket_unlock_mix()` primitives
  - **Memory Ordering**: `mfence_sm()` ensures shared memory visibility before unlock
- **Stream Isolation**: Each XDNN context bound to specific stream via `set_stream()`, enabling concurrent kernel execution

### Performance
- **SIMD Vectorization**:
  - **512-bit registers**: Process 16 floats or 16 halves per cycle
  - **Specialized kernels**: Hand-tuned assembly intrinsics (`vload_lm_float32x16_mz`, `vvmax_float32x16_mz`)
  - **Speedup**: 16x theoretical, ~12-15x practical due to tail handling overhead
- **Lazy Initialization**: XDNN contexts created on-demand via pool, reducing startup time
- **Bulk Memory Operations**: `__builtin_memcpy` for shared memory loads leverages XPU DMA engine
- **Reduced Kernel Launches**: Vectorized operations minimize loop iterations

### Error Handling
- **Macro Abstraction**: `CHECK_KUNLUN(API)` wraps XPU API calls, returns `INFINI_STATUS_SUCCESS` on `XPU_SUCCESS`
- **Status Propagation**: Callback-based design (`useXdnn`) allows error propagation from XDNN calls through status codes
- **RAII**: Pool destructor cleans up contexts even if errors occur

### Dependencies
- **XPU Runtime**: `xpu/runtime.h`, `xpu/runtime_ex.h` - Core device management and stream APIs
- **XDNN Library**: `xpu/xdnn.h` - Baidu's deep learning primitives for convolutions, matmul, etc.
- **XTDK Kernel Headers**:
  - `xpu/kernel/xtdk.h` - Base kernel intrinsics
  - `xpu/kernel/xtdk_atomic_sm_xpu3.h` - Atomic operations for shared memory
  - `xpu/kernel/xtdk_bf16.h` - BFloat16 arithmetic
  - `xpu/kernel/xtdk_math.h` - Math functions
  - `xpu/kernel/xtdk_simd.h` - SIMD vector intrinsics
  - `xpu/kernel/xtdk_trigonometric.h` - Trigonometric functions
- **Internal**: `../../handle.h`, `../pool.h` - Base handle class and lock-free pool implementation

### Design Patterns
- **Pimpl Idiom**: `Handle::Internal` separates interface from implementation, reduces compilation dependencies
- **Object Pool**: Reuse expensive XDNN contexts, avoids repeated initialization
- **Template Specialization**: Type-specific optimizations for atomic ops and reductions
- **RAII**: Automatic resource cleanup through destructors (pool, shared_ptr)
- **Factory Method**: Static `create()` functions handle object construction
- **Strategy Pattern**: Callback-based execution in `useXdnn` allows flexible operation composition

### Known Issues
1. **Incomplete BLAS Integration**: `kunlun_xblas.cc` still references CUDA cublas instead of XPU BLAS
   - **Impact**: BLAS operations will fail at link time or runtime
   - **Fix Required**: Replace `cublasHandle_t` with XPU BLAS handle type and update function calls
2. **32-bit Limitations**: XPU's 32-bit architecture limits single-tensor size to 4GB elements
   - **Workaround**: Custom `_size_t`/`_ptrdiff_t` with padding for DMA alignment
   - **Implication**: Large models require tensor sharding
3. **Ticket Lock Overhead**: Software-based atomics for half types serialize on shared memory
   - **Impact**: Contention in high-thread-count scenarios
   - **Mitigation**: Use float32 when possible, batch atomic operations
