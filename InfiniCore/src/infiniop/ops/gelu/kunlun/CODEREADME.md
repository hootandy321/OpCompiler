# GELU Operation Kunlun Backend Implementation Documentation

This module implements the Gaussian Error Linear Unit (GELU) activation function for Kunlun XPU devices. It leverages the elementwise operation framework to provide optimized GELU computation on Kunlun hardware architecture, supporting BF16, FP16, and FP32 data types with efficient memory access patterns and vectorized operations.

## 1. Module Structure

- **`gelu_kunlun.h`**: Header file defining the GELU descriptor class through the ELEMENTWISE_DESCRIPTOR macro, establishing the public API interface
- **`gelu_kunlun.xpu`**: Implementation file containing descriptor lifecycle management (create/calculate methods) and type-dispatched kernel launching logic
- **`kernel.h`**: Device-side kernel functor implementing the GELU mathematical formula with type-specific optimizations for BF16, FP16, and FP32

## 2. Core Classes

### `op::gelu::kunlun::Descriptor`
- **Location**: `gelu_kunlun.h` (macro-generated), `gelu_kunlun.xpu` (implementation)
- **Primary Function**: Manages GELU operation descriptor for Kunlun XPU devices, encapsulating tensor metadata, device information, workspace requirements, and providing the execution interface
- **Key Members**:
  - `_dtype`: `infiniDtype_t` - Stores the data type (BF16/F16/F32) for type-dispatched kernel execution
  - `_info`: `op::elementwise::ElementwiseInfo` - Encapsulates tensor shapes, strides, broadcast flags, and contiguity metadata for all input/output tensors
  - `_device_info`: `std::unique_ptr<op::elementwise::kunlun::DeviceImpl>` - Opaque pointer to the Kunlun device implementation handling kernel execution
  - `_workspace_size`: `size_t` - Pre-calculated workspace size in bytes required for metadata transfer and kernel execution
- **Core Methods**:
  - `create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, std::vector<infiniopTensorDescriptor_t> input_desc_vec)`: Factory method that validates tensor descriptors (dtype and shape compatibility), constructs ElementwiseInfo, creates DeviceImpl, calculates workspace requirements, and instantiates the descriptor. Returns INFINI_STATUS_SUCCESS on success or error codes for validation failures.
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const`: Executes the GELU operation by verifying workspace sufficiency, dispatching to type-specific kernel instantiation (_dtype switch), and invoking DeviceImpl::calculate with template parameters `<8, GeluOp, T>` where BLOCK_SIZE=8 and T is the concrete type (bfloat16_t/half/float).
  - `~Descriptor()`: Default destructor handling automatic cleanup of managed resources through smart pointers and RAII patterns
- **Lifecycle**: Instantiated via static `create()` factory method with validation; owns DeviceImpl via unique_ptr; destroyed via default destructor when reference count reaches zero

### `op::gelu::kunlun::GeluOp`
- **Location**: `kernel.h`
- **Primary Function**: Device-side functor implementing the GELU activation formula: `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, with type-aware conversions for low-precision arithmetic
- **Key Members**:
  - `num_inputs`: `static constexpr size_t = 1` - Compile-time constant indicating single-input operation
- **Core Methods**:
  - `operator()(const T *x) const __device__`: Device-callable functor that computes GELU activation. Uses if-constexpr for compile-time type dispatch:
    - For `bfloat16_t`: Converts input to float via `__bfloat162float()`, computes GELU in float precision, converts result back via `__float2bfloat16()`
    - For `half`: Converts input to float via `__half2float()`, computes GELU in float precision, converts result back via `__float2half()`
    - For `float`: Direct computation using `fast_erf()` from XPU SDK with formula `0.5 * x * (1 + fast_erf(x / sqrt(2.0f)))`
    - Fallback for other types uses double-precision `sqrt(2.0)`
- **Lifecycle**: Stateless functor instantiated directly in kernel launch; no construction/destruction overhead; exists only as compile-time template parameter

### `op::elementwise::kunlun::DeviceImpl`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun.h` (dependency)
- **Primary Function**: Opaque implementation class managing Kunlun device-side execution, including metadata transfer to device memory, kernel grid configuration, and asynchronous execution orchestration
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl-idiom internal implementation hiding Kunlun-specific details
  - `Opaque::internal`: `std::shared_ptr<device::kunlun::Handle::Internal>` - Raw Kunlun device handle for low-level operations
- **Core Methods**:
  - `create(Args &&...args)`: Factory method constructing DeviceImpl with forwarded arguments to Opaque constructor
  - `calculate<BLOCK_SIZE, Op, Tdata>(ElementwiseInfo, workspace, output, inputs, stream, args...)`: Template method dispatching to `calculateImpl<BLOCK_SIZE, N, Op, Tdata>` where N=Op::num_inputs
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(...)`: Invokes `launchElementwiseKernel()` with kernel function pointer and forwarded arguments
  - `infoToDevice<N>(...)`: Transfers host-side tensor metadata (shapes, strides, contiguity flags, broadcast flags) and input pointer array to device workspace via `xpu_memcpy_async()`, computes device-side pointer offsets using packed metadata layout
  - `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout>(...)`: Configures kernel launch parameters `<<<BLOCK_SIZE, 64, stream>>>` (clusters=BLOCK_SIZE, threads per cluster=64), handles empty tensor early-return, calls kernel function pointer with all metadata and arguments
- **Lifecycle**: Created via static factory; ownership managed through shared_ptr in Opaque pattern; destroyed when last Descriptor reference released

## 3. API Interface

```cpp
namespace op::gelu::kunlun {

// Factory method for descriptor creation
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                    // Kunlun device handle
    Descriptor **desc_ptr,                      // Output parameter for created descriptor
    infiniopTensorDescriptor_t out_desc,        // Output tensor descriptor
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // Single input tensor descriptor
);
// Returns: INFINI_STATUS_SUCCESS on success, INFINI_STATUS_BAD_TENSOR_DTYPE for unsupported types,
//          or error codes from shape/dtype validation

// Execute GELU computation
infiniStatus_t Descriptor::calculate(
    void *workspace,            // Device workspace pointer (size >= workspaceSize())
    size_t workspace_size,      // Actual workspace size in bytes
    void *output,               // Device pointer to output tensor
    std::vector<const void *> inputs,  // Vector of device input pointers (single element)
    void *stream                // Kunlun stream for asynchronous execution
) const;
// Returns: INFINI_STATUS_SUCCESS on success, INFINI_STATUS_INSUFFICIENT_WORKSPACE if workspace too small,
//          INFINI_STATUS_BAD_TENSOR_DTYPE for invalid dtype dispatch

// Query workspace requirements
size_t Descriptor::workspaceSize() const;
// Returns: Pre-calculated workspace size in bytes (metadata + input pointer array)

} // namespace op::gelu::kunlun
```

## 4. Usage Example

```cpp
// Example: GELU activation on Kunlun XPU
#include "infiniop/ops/gelu/kunlun/gelu_kunlun.h"

// Assume we have initialized Kunlun handle and tensors
infiniopHandle_t handle;      // Initialized Kunlun device handle
infiniopTensorDescriptor_t input_desc;   // Input tensor descriptor (shape: {batch_size, seq_len, hidden_dim})
infiniopTensorDescriptor_t output_desc;  // Output tensor descriptor (same shape as input)

// Create GELU descriptor
op::gelu::kunlun::Descriptor* gelu_desc = nullptr;
infiniStatus_t status = op::gelu::kunlun::Descriptor::create(
    handle,
    &gelu_desc,
    output_desc,
    {input_desc}  // Single input
);
if (status != INFINI_STATUS_SUCCESS) {
    // Handle creation error (dtype mismatch or shape mismatch)
}

// Allocate workspace on device
size_t workspace_bytes = gelu_desc->workspaceSize();
void* d_workspace = nullptr;
xpu_malloc(&d_workspace, workspace_bytes);

// Allocate device memory for input/output tensors
void* d_input = nullptr;
void* d_output = nullptr;
xpu_malloc(&d_input, input_desc->size() * sizeof_dtype(input_desc->dtype()));
xpu_malloc(&d_output, output_desc->size() * sizeof_dtype(output_desc->dtype()));

// Copy input data to device (assuming h_input is host pointer)
xpu_memcpy_async(d_input, h_input, input_size_bytes, XPU_HOST_TO_DEVICE, stream);

// Execute GELU operation
status = gelu_desc->calculate(
    d_workspace,      // Device workspace
    workspace_bytes,  // Workspace size
    d_output,         // Device output pointer
    {d_input},        // Device input pointers
    stream            // Kunlun stream
);
if (status != INFINI_STATUS_SUCCESS) {
    // Handle execution error
}

// Copy result back to host (h_output is host pointer)
xpu_memcpy_async(h_output, d_output, output_size_bytes, XPU_DEVICE_TO_HOST, stream);
xpu_synchronize(stream);

// Cleanup (RAII handles descriptor destruction)
delete gelu_desc;
xpu_free(d_workspace);
xpu_free(d_input);
xpu_free(d_output);
```

## 5. Implementation Details

**Mathematical Formula**:
The GELU activation is computed using the standard formulation: `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where `erf()` is the error function. The implementation uses `fast_erf()` from XPU SDK (`xpu/kernel/xtdk_math.h`) which provides an optimized approximation suitable for deep learning workloads, trading minimal accuracy loss for significant performance improvement.

**Memory Management**:
- Workspace layout: Packed structure containing `[input_ptr_array][metadata_blob]` where input_ptr_array occupies `N * sizeof(void*)` bytes (N=1 for GELU) and metadata_blob contains contiguous arrays of output_shape, output_strides, input_shapes, input_strides, input_contiguous_flags, input_broadcasted_flags
- Metadata transfer: Single batch `xpu_memcpy_async()` copies entire metadata blob from host to device workspace, minimizing PCI-e transfer overhead
- Type conversion strategy: BF16 and FP16 inputs are promoted to float32 for GELU computation to maintain numerical stability, then demoted back to original dtype
- Pointer alignment: All device pointers use `__global_ptr__` qualifier for XPU3 architecture optimization

**Concurrency**:
- Kernel launch configuration: `<<<BLOCK_SIZE, 64, stream>>>` where BLOCK_SIZE=8 clusters, 64 threads per cluster (512 total threads)
- Work distribution: Strided loop with each thread processing `len_per_loop = min(64, roundup_div(output_size, nthreads))` elements per iteration to balance load
- Memory access: Local memory buffers (`__local__`) cache metadata and input pointers to reduce global memory access latency
- Synchronization: `mfence()` ensures memory ordering after `GM2LM_ASYNC()` and `LM2GM_ASYNC()` transfers; `sync_cluster()` provides cluster-level barrier

**Performance**:
- Block size tuning: BLOCK_SIZE=8 empirically chosen for GELU workload characteristics balancing occupancy and resource utilization
- Vectorization: Potential for SIMD optimization in float32 path using XPU's 512-bit vector registers (float32x16_t), though current scalar implementation prioritizes correctness
- Loop unrolling: `#pragma unroll` on input copy loop (line 71) enables compiler optimization for fixed N=1 case
- Contiguous fast-path: Index calculation bypasses `indexToOffset()` for contiguous tensors, reducing per-element overhead

**Error Handling**:
- Dtype validation: `CHECK_DTYPE` macro ensures only BF16, FP16, FP32 accepted; returns `INFINI_STATUS_BAD_TENSOR_DTYPE` for unsupported types
- Shape validation: `CHECK_SAME_SHAPE` macro enforces input/output shape congruence; broadcasts not supported for GELU
- Workspace validation: Runtime check `workspace_size < _workspace_size` returns `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- Result type propagation: ElementwiseInfo::create() returns `utils::Result<ElementwiseInfo>` with detailed error codes for descriptor construction failures

**Dependencies**:
- XPU SDK: `xpu/runtime.h`, `xpu/kernel/xtdk.h`, `xpu/kernel/xtdk_bf16.h`, `xpu/kernel/xtdk_math.h` for device intrinsics and math functions
- InfiniOP framework: `elementwise/elementwise.h` for ELEMENTWISE_DESCRIPTOR macro, `devices/kunlun/kunlun_common.h` for common definitions, `devices/kunlun/kunlun_handle.h` for device handle abstraction
- C++ Standard Library: `std::vector`, `std::unique_ptr`, `std::shared_ptr` for RAII resource management

**Design Patterns**:
- Macro-based code generation: `ELEMENTWISE_DESCRIPTOR(gelu, kunlun)` expands to complete Descriptor class definition, eliminating boilerplate across elementwise operations
- Template metaprogramming: `if constexpr` enables type-specific optimizations without runtime branching; CRTP-style kernel functors
- Pimpl idiom: DeviceImpl::Opaque hides Kunlun-specific implementation details from public API
- RAII: Smart pointers (`unique_ptr`, `shared_ptr`) manage device resource lifetimes automatically
- Strategy pattern: Type-dispatched kernel selection via switch statement on `_dtype` enum
