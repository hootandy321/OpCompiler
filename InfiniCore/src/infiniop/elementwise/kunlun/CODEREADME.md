# Kunlun Elementwise Operations Core Implementation Documentation

This module provides the Kunlun XPU (Xeon Phi Processor Unit) backend implementation for elementwise operations in the Infini framework. It handles efficient execution of per-element mathematical operations across tensors with support for broadcasting, strided memory layouts, and flexible data types through kernel-based parallel processing.

## 1. Module Structure

- **`elementwise_kunlun_api.h`**: Defines the public `DeviceImpl` class interface and provides the `CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR` macro for descriptor creation
- **`elementwise_kunlun.h`**: Contains the complete implementation including kernel functions, device memory management, indexing logic, and template-based operation dispatching

## 2. Core Classes

### `DeviceImpl`
- **Location**: `elementwise_kunlun_api.h`
- **Primary Function**: Public facade for Kunlun elementwise operations, providing type-erased interface for launching kernels with template-based polymorphism
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl idiom implementation hiding all Kunlun-specific implementation details
- **Core Methods**:
  - `create(Args&&...args)`: Static factory method returning `Result<DeviceImpl*>` for safe construction with error handling
  - `calculate<BLOCK_SIZE, Op, Tdata, Args...>(info, workspace, output, inputs, stream, args...)`: Main entry point that dispatches to the opaque implementation with template parameters for block size, operation functor, data type, and additional kernel arguments
- **Lifecycle**: Managed by `std::shared_ptr<Opaque>` using pimpl pattern, ensures proper RAII and binary interface stability

### `DeviceImpl::Opaque`
- **Location**: `elementwise_kunlun.h` (lines 161-293)
- **Primary Function**: Encapsulates all Kunlun-specific implementation details including kernel launching, device memory transfers, and metadata management
- **Key Members**:
  - `internal`: `std::shared_ptr<device::kunlun::Handle::Internal>` - Handle to Kunlun device context for XPU operations
- **Core Methods**:
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata, Args...>(info, workspace, output, inputs, stream, args...)`: Type-aware wrapper that forwards to `launchElementwiseKernel` with explicit operation input count `N` verification
  - `infoToDevice<N>(info, workspace, h_inputs_arr, d_inputs_arr, ...)`: Transfers host-side metadata (shapes, strides, contiguous/broadcast flags) to device global memory, computes device pointer offsets through pointer arithmetic on workspace buffer
  - `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout, Args...>(info, workspace, output, inputs, kernel_func, stream, args...)`: Orchestrates kernel execution by validating output size, calling `infoToDevice` for metadata transfer, and launching XPU kernel with triple-angle bracket syntax `<<<BLOCK_SIZE, 64, stream>>>`
- **Lifecycle**: Owned by `DeviceImpl` through shared_ptr, constructed with device handle, manages no external resources

### `InputIndexer`
- **Location**: `elementwise_kunlun.h` (lines 22-36)
- **Primary Function**: Functor that computes memory offsets for input tensors during broadcasting operations, handling both contiguous and strided layouts
- **Key Members**:
  - `idx`: `int` - Current linear index in output tensor
  - `ndim`: `int` - Number of tensor dimensions
  - `input_contiguous`: `const bool*` - Array indicating if each input is contiguous (fast path)
  - `input_broadcasted`: `const bool*` - Array indicating if each input requires broadcasting
  - `input_shapes`: `const _size_t*` - Flattened array of input shapes (N × ndim)
  - `input_strides`: `const _ptrdiff_t*` - Flattened array of input strides (N × ndim)
  - `output_strides`: `const _ptrdiff_t*` - Output tensor strides for dimension mapping
- **Core Methods**:
  - `operator()(input_id)`: Returns memory offset for specified input, uses direct linear index for contiguous tensors or calls `indexToOffset` for strided layouts
- **Complexity**: O(1) for contiguous tensors, O(ndim) for strided tensors per element

## 3. API Interface

```cpp
// Factory method for DeviceImpl creation
template <typename... Args>
static utils::Result<DeviceImpl *> create(Args &&...args);
// Returns heap-allocated DeviceImpl on success, error code on failure

// Main calculation interface with same-type operands
template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t calculate(
    const op::elementwise::ElementwiseInfo &info,    // Tensor metadata (shapes, strides, flags)
    void *workspace,                                  // Device memory for metadata + input pointers
    void *output,                                     // Output tensor device pointer
    const std::vector<const void *> &inputs,          // Input tensor device pointers
    void *stream,                                     // Kunlun stream for async execution
    Args &&...args);                                  // Additional kernel arguments (e.g., scalar values)

// Descriptor creation macro (used in C API layer)
#define CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)
// Automates descriptor construction: creates ElementwiseInfo, computes workspace size,
// instantiates DeviceImpl, and packages into Descriptor object
```

## 4. Usage Example

```cpp
// Example: Launching elementwise addition on Kunlun XPU
// Assuming tensors are already allocated on device

// 1. Create elementwise operation metadata
auto info_result = op::elementwise::ElementwiseInfo::create(output_desc, {input1_desc, input2_desc});
if (!info_result.ok()) { /* handle error */ }
auto info = info_result.take();

// 2. Create Kunlun device implementation
auto device_impl_result = op::elementwise::kunlun::DeviceImpl::create(handle->internal());
if (!device_impl_result.ok()) { /* handle error */ }
auto* device_impl = device_impl_result.take();

// 3. Allocate workspace for metadata (shapes, strides, flags) + input pointer array
size_t workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void*);
void* workspace;
xpu_malloc(&workspace, workspace_size);

// 4. Define operation functor (must satisfy Op::num_inputs concept)
struct AddOp {
    static constexpr int num_inputs = 2;
    template<typename T>
    __device__ T operator()(T* inputs) {
        return inputs[0] + inputs[1];
    }
};

// 5. Launch kernel with 128 block size, float32 data type
infiniStatus_t status = device_impl->calculate<128, AddOp, float>(
    info,
    workspace,
    output_device_ptr,
    {input1_device_ptr, input2_device_ptr},
    kunlun_stream
);

// 6. Synchronize if needed
xpu_stream_synchronize(kunlun_stream);
```

## 5. Implementation Details

### Memory Management
- **Workspace Allocation**: Single contiguous buffer stores both input pointer array (N pointers) followed by metadata blob (shapes, strides, contiguous/broadcast flags) arranged as `[output_shape][output_strides][input_shapes][input_strides][contiguous_flags][broadcast_flags]`
- **Transfer Strategy**: Asynchronous memory transfer via `xpu_memcpy_async` with `XPU_HOST_TO_DEVICE` direction, enables overlap with other operations
- **Pointer Encoding**: Device pointers are encoded as offsets from workspace base through `reinterpret_cast` to `__global_ptr__` (Kunlun's global memory address space qualifier)

### Concurrency
- **Kernel Launch**: XPU kernel launched with `<<<BLOCK_SIZE, 64, stream>>>` where BLOCK_SIZE is template parameter (typically 128), 64 is cluster size, stream specifies execution queue
- **Work Distribution**: Thread ID computed as `thread_id = ncores * cluster_id() + cid` where `ncores` is cores per cluster, `cluster_id()` returns cluster index, `cid` is core ID within cluster
- **Loop Tiling**: Processes `BUFF_SIZE=64` elements per loop iteration with `min(BUFF_SIZE, roundup_div(output_size, nthreads))` chunking for load balancing
- **Synchronization**: `mfence()` after `GM2LM_ASYNC` (global-to-local memory) and `LM2GM_ASYNC` (local-to-global memory) ensures memory coherence, `sync_cluster()` at kernel end for cluster-wide barrier

### Performance
- **Contiguous Fast Path**: When `input_contiguous[i]` is true, uses direct linear index (`idx`) avoiding `indexToOffset` computation, eliminating O(ndim) per-element overhead
- **Local Memory Caching**: Uses `__local__` memory (on-chip scratchpad) for frequently accessed metadata (shapes, strides, flags) loaded once at kernel start via `GM2LM_ASYNC` bulk transfers
- **Loop Unrolling**: `#pragma unroll` on input copy loop (line 71) enables compiler optimization for fixed iteration count N (operation input count)
- **Chunking Strategy**: `len_per_loop = min(BUFF_SIZE, roundup_div(output_size, nthreads))` balances chunk size for memory bandwidth vs parallelism

### Error Handling
- **Result Type**: Factory method returns `utils::Result<DeviceImpl*>` forcing explicit error checking with `.ok()` and `.take()`
- **Status Codes**: All functions return `infiniStatus_t` enum, `INFINI_STATUS_SUCCESS` indicates success
- **CHECK Macros**: `CHECK_KUNLUN` and `CHECK_STATUS` macros wrap XPU API calls and internal functions, propagating errors on failure
- **Zero-Size Guard**: `launchElementwiseKernel` returns early if `output_size == 0` to avoid invalid kernel launches

### Dependencies
- **External**: Kunlun XPU runtime (`xpu_memcpy_async`, `xpu_stream_synchronize`), XPU device headers (`kunlun_common.h`, `kunlun_handle.h`, `kunlun_kernel_common.h`)
- **Internal**: `op::elementwise::ElementwiseInfo` for metadata management, `device::kunlun::Handle::Internal` for device context, parent `elementwise.h` for base definitions
- **Inter-Module**: Depends on Kunlun device abstraction layer (`device::kunlun`), elementwise common utilities (`op::elementwise`)

### Design Patterns
- **Pimpl (Pointer to Implementation)**: `DeviceImpl` forwards all operations to `DeviceImpl::Opaque`, hiding Kunlun-specific types in `.h` file from public API
- **Template-based Polymorphism**: Operation functors specified as template parameter `Op` with `num_inputs` static member enables compile-time specialization without virtual functions
- **CRTP-style Kernel Functors**: `Op` functors called as `Op{}(inputs_buf, args...)` in device code, allowing stateless operation objects with `operator()` overloads
- **Type Erasure**: Public `calculate` interface uses `void*` for tensor pointers while internal implementation uses strongly-typed `__global_ptr__ Tdata*` for type-safe kernel access
