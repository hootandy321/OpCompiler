# SwiGLU METAX Operator Core Implementation Documentation

This module implements the SwiGLU (Swish-Gated Linear Unit) activation function for Moore Threads METAX GPU accelerators. SwiGLU is a powerful activation function widely used in modern transformer architectures like LLaMA, combining SiLU activation with gated linear units for enhanced expressiveness.

## 1. Module Structure

- **`swiglu_metax.h`**: API header defining the Descriptor class through the ELEMENTWISE_DESCRIPTOR macro
- **`swiglu_metax.maca`**: Implementation file with descriptor creation, validation, and kernel dispatch logic

## 2. Core Classes

### `op::swiglu::metax::Descriptor`
- **Location**: `swiglu_metax.h`, `swiglu_metax.maca`
- **Primary Function**: Manages SwiGLU operation lifecycle for METAX devices, including descriptor creation, workspace management, and kernel execution
- **Key Members**:
  - `_dtype`: `infiniDtype_t` - Supported data type (F16, BF16, F32, F64)
  - `_info`: `op::elementwise::ElementwiseInfo` - Tensor metadata including shapes, strides, contiguity flags, and broadcast information
  - `_device_info`: `std::unique_ptr<op::elementwise::metax::DeviceImpl>` - METAX device-specific implementation wrapper
  - `_workspace_size`: `size_t` - Required workspace memory in bytes for metadata and input pointer arrays
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: Factory method that validates tensor descriptors, extracts metadata, and constructs the descriptor with appropriate workspace allocation
  - `calculate(workspace, workspace_size, output, inputs, stream)`: Executes the SwiGLU kernel on METAX device, dispatching to dtype-specific implementations
  - `~Descriptor()`: Default destructor
- **Lifecycle**: Singleton-per-operation pattern. Created via `create()` factory method, manages its own device implementation instance, destroyed when no longer referenced

### `op::swiglu::cuda::SwiGLUOp` (Functor)
- **Location**: `../cuda/kernel.cuh` (referenced, not in current directory)
- **Primary Function**: Device-side functor implementing the SwiGLU mathematical operation: `output = SiLU(gate) * up`, where `SiLU(x) = x * sigmoid(x)`
- **Key Members**:
  - `num_inputs`: `constexpr size_t = 2` - Specifies two input tensors (up and gate)
- **Core Methods**:
  - `sigmoid(x)`: Private helper computing sigmoid activation with type-optimized implementations for half2, half, cuda_bfloat162, cuda_bfloat16, float, and double
  - `operator()(up, gate)`: Public call operator returning `gate * sigmoid(gate) * up` with vectorized intrinsics for half2 (h2rcp, h2exp, __hneg2) and cuda_bfloat162 types
- **Design**: Compile-time polymorphism using `if constexpr` for zero-runtime-overhead type dispatch

### `op::elementwise::metax::DeviceImpl`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/metax/elementwise_metax.h` (parent module)
- **Primary Function**: Generic elementwise operation engine for METAX devices, handling kernel launch, memory management, and grid/block configuration
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>` - PIMPL pointer to internal implementation holding device handle
- **Core Methods**:
  - `create(args...)`: Factory constructing DeviceImpl with shared ownership of internal state
  - `calculate<BLOCK_SIZE, Op, Tdata>(info, workspace, output, inputs, stream, args...)`: Template method dispatching to type-specific kernel implementation with 256-thread block size
  - `calculate<BLOCK_SIZE, Op, Tout, Tin...>(info, workspace, output, inputs, stream, args...)`: Overload supporting mixed input/output types
- **Internal Architecture**:
  - `Opaque::calculateImpl()`: Validates workspace size, copies metadata to GPU via hcMemcpyAsync, configures kernel grid dimensions
  - `Opaque::infoToDevice()`: Packs tensor metadata (shapes, strides, contiguity flags) into device workspace using pointer arithmetic offsets
  - `Opaque::launchElementwiseKernel()`: Calculates grid/block dimensions respecting device limits (maxThreadsPerBlock, gridSizeX), launches kernel in grid-stride loops for large outputs

### `op::elementwise::ElementwiseInfo`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h` (parent module)
- **Primary Function**: Immutable metadata container storing tensor layout information for both inputs and outputs
- **Key Members**:
  - `_meta`: `std::vector<size_t>` - Packed binary layout containing: output_shape, output_strides, all input_shapes, all input_strides, input_contiguous flags, input_broadcasted flags
  - `_output_size`: `size_t` - Total number of elements in output tensor
  - `_input_size`: `size_t` - Number of input tensors (always 2 for SwiGLU)
  - `_ndim`: `size_t` - Tensor rank/number of dimensions
  - `_output_contiguous`: `bool` - Whether output has row-major contiguous layout
- **Core Methods**:
  - `create(output_desc, input_descs)`: Static factory validating descriptors, computing metadata memory requirements, packing all tensor properties into single contiguous buffer
  - `getMetaMemSize()`: Returns packed metadata size in bytes
  - `getOutputShape()`, `getOutputStrides()`, `getInputShape(index)`, `getInputStrides(index)`: Accessors returning typed pointers into packed metadata
  - `getInputContiguous()`, `getInputBroadcasted()`: Boolean arrays indicating per-input layout properties
- **Memory Layout**: Single allocation strategy using `std::vector<size_t>` with manual pointer casting to access different sections, avoiding multiple allocations

## 3. API Interface

```cpp
namespace op::swiglu::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    // Factory method - constructs descriptor with validated metadata
    static infiniStatus_t create(
        infiniopHandle_t handle_,                  // METAX device handle
        Descriptor **desc_ptr,                      // Output parameter for created descriptor
        infiniopTensorDescriptor_t out_desc,        // Output tensor descriptor
        std::vector<infiniopTensorDescriptor_t> input_desc_vec); // {up, gate} descriptors

    // Execute SwiGLU computation
    infiniStatus_t calculate(
        void *workspace,              // Device workspace for metadata (size >= workspaceSize())
        size_t workspace_size,        // Must equal workspaceSize()
        void *output,                 // Device pointer to output tensor
        std::vector<const void *> inputs, // {up_device_ptr, gate_device_ptr}
        void *stream) const;          // hcStream_t for METAX async execution

    // Query required workspace size
    size_t workspaceSize() const;
};

} // namespace op::swiglu::metax
```

## 4. Usage Example

```cpp
#include "swiglu_metax.h"
#include <vector>

// Example: Computing SwiGLU activation in a transformer feed-forward network
void example_swiglu_metax_usage() {
    // 1. Initialize METAX device handle
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);

    // 2. Create tensor descriptors (assuming shape [batch_size, seq_len, hidden_dim])
    std::vector<int64_t> shape = {32, 2048, 4096};
    std::vector<int64_t> strides = {8388608, 4096, 1}; // Contiguous row-major

    infiniopTensorDescriptor_t up_desc, gate_desc, out_desc;
    infoniopCreateTensorDescriptor(&up_desc, INFINI_DTYPE_F16, shape.data(), strides.data(), 3);
    infoniopCreateTensorDescriptor(&gate_desc, INFINI_DTYPE_F16, shape.data(), strides.data(), 3);
    infoniopCreateTensorDescriptor(&out_desc, INFINI_DTYPE_F16, shape.data(), strides.data(), 3);

    // 3. Create SwiGLU descriptor
    op::swiglu::metax::Descriptor *swiglu_desc = nullptr;
    auto status = op::swiglu::metax::Descriptor::create(
        handle,
        &swiglu_desc,
        out_desc,
        {up_desc, gate_desc});

    if (status != INFINI_STATUS_SUCCESS) {
        // Handle error (bad dtype, shape mismatch, etc.)
        return;
    }

    // 4. Allocate device memory
    size_t tensor_bytes = 32 * 2048 * 4096 * sizeof(half); // ~1GB
    half *d_up, *d_gate, *d_out;
    hcMalloc((void**)&d_up, tensor_bytes);
    hcMalloc((void**)&d_gate, tensor_bytes);
    hcMalloc((void**)&d_out, tensor_bytes);

    size_t workspace_size = swiglu_desc->workspaceSize();
    void *d_workspace;
    hcMalloc(&d_workspace, workspace_size);

    // 5. Copy input data to device (assuming h_up, h_gate are host arrays)
    hcMemcpyAsync(d_up, h_up, tensor_bytes, hcMemcpyHostToDevice, stream);
    hcMemcpyAsync(d_gate, h_gate, tensor_bytes, hcMemcpyHostToDevice, stream);

    // 6. Execute SwiGLU kernel
    status = swiglu_desc->calculate(
        d_workspace,
        workspace_size,
        d_out,
        {d_up, d_gate},
        stream);

    // 7. Copy result back to host
    hcMemcpyAsync(h_out, d_out, tensor_bytes, hcMemcpyDeviceToHost, stream);
    hcStreamSynchronize(stream);

    // 8. Cleanup
    delete swiglu_desc;
    hcFree(d_up); hcFree(d_gate); hcFree(d_out); hcFree(d_workspace);
    infoniopDestroyTensorDescriptor(up_desc);
    infoniopDestroyTensorDescriptor(gate_desc);
    infoniopDestroyTensorDescriptor(out_desc);
    infiniopDestroyHandle(handle);
}
```

## 5. Implementation Details

### Memory Management
- **Packed Metadata Strategy**: All tensor properties (shapes, strides, contiguity) packed into single `std::vector<size_t>` via manual pointer offsets, minimizing allocations and enabling single GPU memcpy for metadata transfer
- **Workspace Layout**: Device workspace organized as:
  1. Input pointer array (2 × sizeof(void*)) for {up, gate}
  2. Metadata block starting at offset `input_arr_size` containing output_shape, output_strides, input_shapes, input_strides, contiguity flags, broadcast flags
- **Shared Ownership**: DeviceImpl uses `std::shared_ptr<Opaque>` for safe concurrent access and automatic cleanup

### Concurrency
- **Asynchronous Execution**: All GPU operations use `hcMemcpyAsync` and kernel launches with `hcStream_t`, enabling overlapping compute/data transfers
- **Thread Safety**: Descriptor instances are thread-safe for read-only concurrent execution (calculate() is const method), but create() must be synchronized
- **No Internal Locking**: Relies on external stream synchronization; user must manage stream ordering

### Performance
- **Vectorized Intrinsics**: Half precision (FP16) uses half2 packed operations (`__hmul2`, `h2exp`, `h2rcp`) processing 2 values per instruction, doubling throughput
- **BFloat16 Optimization**: Custom implementation using `__bfloat162float` conversions since METAX lacks native bfloat16 arithmetic, minimizing conversion overhead
- **Grid-Stride Loops**: Kernel launched with `gridDims.x * blockDims.x` step size, processing outputs in chunks to exceed maximum grid dimension limits
- **Block Size**: Fixed 256 threads per block (optimal for METAX SIMD width), calculated as `min(256, maxThreadsPerBlock)`
- **Contiguity Optimization**: Fast-path when tensors are contiguous (no index calculation), using linear thread-to-element mapping

### Error Handling
- **Descriptor Creation**:
  - `CHECK_DTYPE`: Validates data type is F16/BF16/F32/F64, returns `INFINI_STATUS_BAD_TENSOR_DTYPE` on mismatch
  - `CHECK_SAME_SHAPE`: Ensures output, up, and gate tensors have identical shapes, returns error on mismatch
  - `CHECK_RESULT`: Propagates ElementwiseInfo construction failures (bad param, broadcast in output)
- **Runtime Validation**:
  - Workspace size checked: returns `INFINI_STATUS_INSUFFICIENT_WORKSPACE` if too small
  - Empty tensor early-exit: returns success immediately if output_size == 0
- **METAX Call Wrapper**: `CHECK_METAX` macro wraps HC API calls, converting error codes to `infiniStatus_t`

### Dependencies
- **Parent Modules**:
  - `op::elementwise::metax::DeviceImpl`: Generic METAX elementwise execution engine
  - `op::elementwise::ElementwiseInfo`: Tensor metadata container
  - `device::metax::Handle`: METAX device abstraction with HC context
- **CUDA Kernel Functor**: Reuses `op::swiglu::cuda::SwiGLUOp` from `../cuda/kernel.cuh` for device computation (compatible due to NVPTX/HC ISA similarities)
- **External Libraries**:
  - HC (Moore Threads HIP Compatibility Layer): Device memory management (`hcMalloc`, `hcMemcpyAsync`), kernel launches (`<<<>>>`)
  - InfiniOP core utilities: `utils::Result<T>` for error handling, tensor descriptor abstraction

### Design Patterns
- **CRTP (Curiously Recurring Template Pattern)**: `ELEMENTWISE_DESCRIPTOR` macro generates Descriptor class inheriting from `InfiniopDescriptor` with boilerplate constructor/members
- **PIMPL (Pointer to Implementation)**: DeviceImpl hides internal Opaque struct, reducing compilation dependencies
- **Factory Method**: `create()` static method handles complex initialization logic, returning error status instead of throwing exceptions
- **Functor Strategy**: SwiGLUOp implements `operator()` for generic elementwise framework invocation
- **Template Metaprogramming**: Compile-time dispatch via `if constexpr` eliminates runtime branching in device code
- **Type Erasure**: `void*` and `std::vector<const void*>` used in public API for dtype-agnostic interface, with type safety restored in template methods
