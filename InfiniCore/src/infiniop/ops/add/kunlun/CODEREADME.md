# Kunlun Add Operation Core Implementation Documentation

This module implements the element-wise addition operation for Kunlun XPU (Xuanwu Architecture) devices, providing high-performance tensor addition with support for broadcasting, non-contiguous memory layouts, and mixed precision arithmetic. The implementation leverages Kunlun's XPU kernel framework with optimized memory access patterns and SIMD vectorization.

## 1. Module Structure

- **`add_kunlun.h`**: Public API header that defines the Descriptor class using the element-wise operation macro
- **`add_kunlun.xpu`**: Main implementation file containing descriptor lifecycle management and kernel dispatch logic
- **`kernel.h`**: Device-side kernel functor defining the addition operation with type-specific specializations

## 2. Core Classes

### `op::add::kunlun::Descriptor`
- **Location**: `add_kunlun.h` (macro-generated), `add_kunlun.xpu` (implementation)
- **Primary Function**: Manages the lifecycle and execution of element-wise addition operations on Kunlun XPU devices, handling validation, workspace management, and kernel dispatch
- **Key Members**:
  - `_dtype`: `infiniDtype_t` - Data type identifier (F16, F32, BF16, I32)
  - `_info`: `op::elementwise::ElementwiseInfo` - Metadata containing shapes, strides, and layout information for output and input tensors
  - `_device_info`: `std::unique_ptr<op::elementwise::kunlun::DeviceImpl>` - Kunlun-specific device implementation for kernel execution
  - `_workspace_size`: `size_t` - Required workspace memory size in bytes for metadata and pointer arrays
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: Static factory method that validates tensor descriptors, creates ElementwiseInfo metadata, initializes the Kunlun device implementation, and computes workspace requirements
  - `calculate(workspace, workspace_size, output, inputs, stream)`: Executes the addition operation by dispatching to the appropriate type-specialized kernel (F16/F32/BF16/I32), validates workspace size, and calls DeviceImpl::calculate with block size 8
- **Lifecycle**: Created via static factory method with validation; ownership transferred to caller; destroyed via default destructor when descriptor is no longer needed

### `op::add::kunlun::AddOp`
- **Location**: `kernel.h`
- **Primary Function**: Device-side functor that performs element-wise addition with two input operands, providing specializations for different data types including precision-preserving bfloat16 arithmetic
- **Key Members**:
  - `num_inputs`: `static constexpr int` = 2 - Compile-time constant indicating binary operation
- **Core Methods**:
  - `operator()(const T *inputs)`: Device-callable functor that extracts two operands from the input array and returns their sum; uses float-precision arithmetic for bfloat16 to maintain computational accuracy
  - `operator()(const bfloat16_t *inputs)`: Specialized version that converts bfloat16 inputs to float precision using `__bfloat162float`, performs addition, then converts back using `__float2bfloat16` to avoid precision loss
- **Lifecycle**: Stateless functor instantiated on-device during kernel execution; no persistent state

### `op::elementwise::kunlun::DeviceImpl`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun.h` (parent implementation)
- **Primary Function**: Encapsulates Kunlun XPU device-side execution logic for element-wise operations, managing metadata transfer to device memory, kernel launch configuration, and memory access optimization
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl pattern hiding device-specific implementation details including handle internal state
- **Core Methods**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(info, workspace, output, inputs, stream)`: Template method that instantiates the kernel with specified block size (default 8 for add), operation functor, and data type; delegates to Opaque::calculateImpl
- **Lifecycle**: Created via static factory with shared_ptr to Handle::Internal; destroyed when no longer referenced

### `op::elementwise::kunlun::DeviceImpl::Opaque`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun.h`
- **Primary Function**: Implementation class managing low-level Kunlun XPU kernel execution, workspace memory layout, and metadata transfer between host and device
- **Key Members**:
  - `internal`: `std::shared_ptr<device::kunlun::Handle::Internal>` - Reference to device handle for accessing XPU runtime and XDNN context
- **Core Methods**:
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(info, workspace, output, inputs, stream, args...)`: Template method that validates output size, transfers metadata to device via infoToDevice, and launches the elementwiseKernel with <<<BLOCK_SIZE, 64, stream>>> configuration
  - `infoToDevice<N>(info, workspace, h_inputs_arr, ...)`: Transfers tensor metadata (shapes, strides, contiguity flags, broadcasting flags) and input pointer array from host to device workspace memory using asynchronous XPU memcpy; calculates device-side pointer offsets for flattened metadata arrays
  - `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout>(...)`: Orchestrates kernel launch by calling infoToDevice to setup metadata, then invokes the kernel function with <<<BLOCK_SIZE, 64, stream>>> execution configuration
- **Lifecycle**: Owned by DeviceImpl via shared_ptr; constructed with Handle::Internal reference

## 3. API Interface

### Public C++ API

```cpp
namespace op::add::kunlun {

// Factory method for creating add operation descriptor
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                    // Kunlun device handle
    Descriptor **desc_ptr,                        // Output parameter for created descriptor
    infiniopTensorDescriptor_t out_desc,          // Output tensor descriptor
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) // Input descriptors [A, B]
// Returns: INFINI_STATUS_SUCCESS on success, error code on validation failure
// Validates: dtype (F16/F32/BF16/I32), shape compatibility, ndim alignment

// Execute addition operation
infiniStatus_t Descriptor::calculate(
    void *workspace,                              // Device workspace memory pointer
    size_t workspace_size,                        // Size of workspace in bytes
    void *output,                                 // Device output tensor pointer
    std::vector<const void *> inputs,             // Device input tensor pointers [A, B]
    void *stream) const                           // Kunlun XPU stream handle
// Returns: INFINI_STATUS_SUCCESS or error code
// Dispatches: Type-specialized kernel based on _dtype member

// Query workspace requirements
size_t Descriptor::workspaceSize() const
// Returns: Required workspace size for metadata and input pointer array
}
```

### Device Kernel Functor Interface

```cpp
namespace op::add::kunlun {

struct AddOp {
    static constexpr int num_inputs = 2;

    // Generic type template
    template <typename T>
    inline __device__ T operator()(const T *inputs) const {
        T a = inputs[0];
        T b = inputs[1];
        return a + b;
    }

    // bfloat16 specialization for precision preservation
    inline __device__ bfloat16_t operator()(const bfloat16_t *inputs) const {
        float a_f = __bfloat162float(inputs[0]);
        float b_f = __bfloat162float(inputs[1]);
        return __float2bfloat16(a_f + b_f);
    }
};
}
```

### Elementwise Metadata Structure

```cpp
struct ElementwiseInfo {
    // Layout of flattened metadata in host/device memory:
    // [output_shape(ndim)]
    // [output_strides(ndim)]
    // [input_shapes(N * ndim)]
    // [input_strides(N * ndim)]
    // [input_contiguous(N)]
    // [input_broadcasted(N)]

    size_t getMetaMemSize() const;           // Total metadata size in bytes
    const int8_t* getMetaStart() const;      // Pointer to metadata array
    size_t getOutputSize() const;            // Total number of elements
    size_t getInputSize() const;             // Number of input tensors (2 for add)
    size_t getNdim() const;                  // Number of dimensions
    bool isOutputContiguous() const;         // Whether output is contiguous

    // Accessors for specific metadata sections
    const size_t* getOutputShape() const;
    const ptrdiff_t* getOutputStrides() const;
    const size_t* getInputShape(size_t index) const;
    const ptrdiff_t* getInputStrides(size_t index) const;
    const bool* getInputContiguous() const;   // Array[N]
    const bool* getInputBroadcasted() const;  // Array[N]
};
```

## 4. Usage Example

```cpp
#include "add_kunlun.h"
#include "infiniop.h"

using namespace op::add::kunlun;

// Initialize Kunlun device handle
infiniopHandle_t handle;
infiniStatus_t status = Handle::create(&handle, 0);  // device_id = 0

// Create tensor descriptors for C = A + B
std::vector<int64_t> shape = {1024, 1024};
infiniopTensorDescriptor_t desc_a, desc_b, desc_c;
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, shape.size(),
                               shape.data(), &desc_a);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, shape.size(),
                               shape.data(), &desc_b);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F32, shape.size(),
                               shape.data(), &desc_c);

// Create add operation descriptor
Descriptor* add_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {desc_a, desc_b};
status = Descriptor::create(handle, &add_desc, desc_c, inputs);

// Allocate device memory
float *d_a, *d_b, *d_c;
size_t nbytes = 1024 * 1024 * sizeof(float);
xpu_malloc(reinterpret_cast<void**>(&d_a), nbytes);
xpu_malloc(reinterpret_cast<void**>(&d_b), nbytes);
xpu_malloc(reinterpret_cast<void**>(&d_c), nbytes);

// Allocate workspace for metadata and input pointers
size_t workspace_size = add_desc->workspaceSize();
void *d_workspace;
xpu_malloc(&d_workspace, workspace_size);

// Create Kunlun stream
kunlunStream_t stream;
xpu_stream_create(&stream);

// Execute addition: C = A + B
std::vector<const void *> input_ptrs = {d_a, d_b};
status = add_desc->calculate(d_workspace, workspace_size, d_c, input_ptrs, stream);

// Synchronize and cleanup
xpu_stream_synchronize(stream);
xpu_free(d_a);
xpu_free(d_b);
xpu_free(d_c);
xpu_free(d_workspace);
xpu_stream_destroy(stream);
delete add_desc;
```

## 5. Implementation Details

### Memory Management

- **Workspace Layout**: Device workspace contains two regions:
  1. Input pointer array (N * sizeof(void*)) at workspace base
  2. Flattened metadata starting at workspace + input_arr_size, organized as [output_shape, output_strides, input_shapes, input_strides, input_contiguous, input_broadcasted]

- **Memory Transfer Strategy**: Uses asynchronous XPU memory transfers (`xpu_memcpy_async`) with XPU_HOST_TO_DEVICE direction for non-blocking metadata and pointer array transfers

- **Local Memory Usage**: Kernel allocates local memory buffers for:
  - `inputs_buf[N]`: Input operand cache (N=2 for add)
  - Metadata arrays: input_contiguous[N], input_broadcasted[N], input_shapes[N*ndim], input_strides[N*ndim], output_shape[ndim], output_strides[ndim]
  - `typed_inputs_ptr[N]`: Cached device pointer array

### Concurrency

- **Thread Hierarchy**: Kunlun XPU uses cluster-based parallel execution:
  - `core_id()`: Unique core ID within a cluster (0 to ncores-1)
  - `cluster_id()`: Cluster identifier
  - `thread_id = ncores * cluster_id() + cid`: Global thread ID
  - Total threads: `ncores * cluster_num()`

- **Kernel Launch Configuration**: <<<BLOCK_SIZE, 64, stream>>>
  - BLOCK_SIZE: Template parameter (8 for add operation) controlling cluster/block count
  - 64: Threads per cluster/cluster size
  - Grid/Cluster dimension automatically calculated by runtime

- **Work Distribution**: Each thread processes `len_per_loop` elements (min of 64 or rounded-up division of total size), iterating with stride of `nthreads * len_per_loop`

- **Memory Fencing**: Uses `mfence()` for local memory consistency after GM2LM_ASYNC/LM2GM_ASYNC operations; `sync_cluster()` at kernel completion

### Performance

- **SIMD Vectorization**: Parent implementation uses 512-bit SIMD registers for float (16 elements) and half (32 elements) operations in reduction operations, though add operation itself is scalar per-element

- **Memory Access Optimization**:
  - GM2LM_ASYNC: Asynchronous global-to-local memory copy with explicit size (1 * sizeof(Tdata) per input)
  - LM2GM_ASYNC: Asynchronous local-to-global memory copy for result writeback
  - Contiguous tensor fast-path: Direct linear indexing when input_contiguous=true
  - Non-contiguous support: `indexToOffset` function computes physical memory offset from linear index using shape/stride arrays

- **Block Size Selection**: Template parameter BLOCK_SIZE=8 balances parallelism with overhead for element-wise operations; allows compile-time optimization

- **Chunked Processing**: BUFF_SIZE=64 limits local memory buffer size, processing output in chunks to fit within local memory constraints while reducing global memory access

### Error Handling

- **Validation Layers**:
  - `CHECK_DTYPE`: Validates dtype is F16, F32, BF16, or I32
  - `CHECK_SAME_SHAPE`: Ensures all tensors (output and inputs) have identical shapes
  - Workspace size check in calculate() returns INFINI_STATUS_INSUFFICIENT_WORKSPACE if workspace too small

- **Error Propagation**: Uses Result<T> pattern for ElementwiseInfo creation; CHECK_RESULT macro unwraps Result or returns error early

- **Status Codes**: Returns infiniStatus_t codes:
  - INFINI_STATUS_SUCCESS: Operation completed successfully
  - INFINI_STATUS_BAD_TENSOR_DTYPE: Unsupported dtype
  - INFINI_STATUS_INSUFFICIENT_WORKSPACE: Workspace too small
  - INFINI_STATUS_BAD_PARAM: Null descriptor pointers
  - INFINI_STATUS_BAD_TENSOR_STRIDES: Invalid stride configuration

### Dependencies

- **External Libraries**:
  - `xpu/runtime.h`: Kunlun XPU runtime API (XPUStream, xpu_malloc, xpu_memcpy_async)
  - `xpu/runtime_ex.h`: Extended runtime functions
  - `xpu/xdnn.h`: Baidu XPU XDNN context (xdnn::Context)

- **Internal Modules**:
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/kunlun/elementwise_kunlun.h`: DeviceImpl class and metadata management
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`: ElementwiseInfo metadata structure and ELEMENTWISE_DESCRIPTOR macro
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/kunlun/kunlun_common.h`: Kunlun type definitions and error checking macros
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/kunlun/kunlun_kernel_common.h`: Device-side utilities (indexToOffset, atomic operations, SIMD functions)
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/kunlun/kunlun_handle.h`: Device handle and XDNN context management

### Design Patterns

- **Macro-Based Code Generation**: ELEMENTWISE_DESCRIPTOR macro generates Descriptor class boilerplate, reducing duplication across element-wise operations (add, sub, mul, etc.)

- **Pimpl Idiom**: DeviceImpl uses Opaque nested class to hide kernel launch and metadata transfer implementation details

- **Functor Pattern**: AddOp struct with operator() overloading enables compile-time polymorphism for different element-wise operations using same kernel template

- **Template Metaprogramming**: Extensive use of template parameters (BLOCK_SIZE, N, Op, Tdata) enables compile-time optimization and code specialization

- **RAII**: std::unique_ptr for DeviceImpl ownership; std::shared_ptr for Opaque and Handle::Internal lifecycle management

- **CRTP (Curiously Recurring Template Pattern)**: Implied through macro expansion where Descriptor inherits from InfiniopDescriptor with operation-specific configuration

- **Policy-Based Design**: Kernel function passed as template parameter to launchElementwiseKernel, allowing runtime selection of operation while maintaining type safety

### Type Specializations

- **FP32 (float)**: Direct addition using native float arithmetic
- **FP16 (half)**: Direct addition using half precision; may have reduced precision but faster execution
- **BF16 (bfloat16_t)**: Precision-preserving specialization that converts to float, performs addition, then converts back to maintain computational accuracy; prevents accumulation of rounding errors
- **INT32 (int32_t)**: Standard integer addition for integer tensors

### Broadcasting and Stride Handling

- **Broadcast Detection**: `input_broadcasted` flag set when input has different ndim or contains broadcast dimensions (stride=0 for broadcasted axes)

- **Index Calculation**: `InputIndexer` functor computes memory offset for each input:
  - Contiguous path: Direct linear index (no stride computation)
  - Non-contiguous path: `indexToOffset` converts linear index to physical offset using nested modulo/division based on shape and stride arrays

- **Shape Broadcasting**: ElementwiseInfo::create validates broadcast compatibility by ensuring output has no broadcast dimensions itself; inputs can be broadcast to match output shape

### Kernel Execution Flow

1. **Setup Phase**:
   - Load cluster info (core_id, cluster_id, ncores, cluster_num)
   - Cast input pointers to typed __global_ptr__ Tdata**
   - Allocate local memory buffers for metadata and input cache

2. **Metadata Transfer**:
   - GM2LM_ASYNC: Copy input_contiguous, input_broadcasted, input_shapes, input_strides, output_shape, output_strides, typed_inputs_ptr from global to local memory
   - mfence(): Ensure all async transfers complete

3. **Compute Loop**:
   - Calculate per-thread work chunk: len_per_loop = min(64, ceil_div(output_size, nthreads))
   - For each chunk: for (idx = start; idx < start + read_len; ++idx)
     - Compute output index (direct or via getOutputIndex for non-contiguous)
     - Compute input indices using InputIndexer for each operand
     - launchOp<2, AddOp, Tdata>: Load 2 inputs via GM2LM_ASYNC, compute AddOp, store result via LM2GM_ASYNC

4. **Synchronization**:
   - sync_cluster(): Ensure all cores in cluster complete before returning
