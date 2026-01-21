# CUDA Kernels Core Implementation Documentation

This directory contains CUDA kernel implementations for InfiniTrain's deep learning operations, providing optimized GPU-accelerated computations for neural network training and inference. All kernels support automatic differentiation with forward and backward passes, covering fundamental operations including matrix multiplication, elementwise computations, reductions, and tensor manipulations.

## 1. Module Structure

- **`accumulate_grad.cu`**: Gradient accumulation and Adam optimizer implementation with bias correction
- **`cast.cu`**: Data type conversion kernels with support for all numeric types
- **`comm.cu`**: Communication primitives for distributed training (broadcast, scatter, gather, reduce)
- **`concat.cu`**: Tensor concatenation along specified dimension with efficient memory access patterns
- **`cross_entropy.cu`**: Cross-entropy loss computation with numerical stability optimizations
- **`elementwise.cu`**: Comprehensive elementwise operations (arithmetic, trigonometric, logical) with broadcasting support
- **`embedding.cu`**: Embedding layer lookup and gradient computation using atomic operations
- **`fill.cu`**: Tensor initialization with constant values
- **`gather.cu`**: Index-based gathering operations (PyTorch-compatible gather)
- **`layernorm.cu`**: Layer normalization with efficient mean/variance computation using CUB block reduction
- **`linear.cu`**: Linear transformation and matrix multiplication using cuBLAS GEMM routines
- **`no_op.cu`**: Zero-copy tensor view operations for memory-efficient reshaping
- **`outer.cu`**: Outer product computation optimized for vector operations
- **`reduction.cu`**: Tensor reduction operations (sum, mean, max, min) with generic backend
- **`slice.cu`**: Tensor slicing with arbitrary start/end/step parameters
- **`softmax.cu`**: Softmax activation with stable exponent computation
- **`split.cu`**: Tensor splitting into equal-sized chunks
- **`stack.cu`**: Tensor stacking along new dimension
- **`transform.cu`**: Tensor transformations (transpose, tril, triu, mask, repeat_interleave)
- **`vocab_parallel_cross_entropy.cu`**: Vocabulary-parallel cross-entropy for distributed training

## 2. Core Classes

### `AccumulateGradKernel<T>`
- **Location**: `accumulate_grad.cu`
- **Primary Function**: Performs gradient accumulation with optional learning rate scaling
- **Key Members**:
  - `grad_ptr`: Pointer to gradient tensor (const T*)
  - `rate`: Learning rate scaling factor (float)
  - `tensor_ptr`: Pointer to parameter tensor to update (T*)
  - `num_elements`: Total number of elements to process (size_t)
- **Core Methods**:
  - `AccumulateGradKernel<<<grid, block>>>(grad_ptr, rate, tensor_ptr, num_elements)`: Element-wise scaled gradient accumulation with O(1) complexity per element
  - `AdamAccumulateGradKernel<<<grid, block>>>(grad_data, param_data, m_data, v_data, num_elements, learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v)`: Adam optimizer update with bias-corrected moments, uses fused multiply-add (FMA) for efficient moment computation
- **Lifecycle**: Stateless kernel launched with 256 threads per block, grid size computed as CEIL_DIV(num_elements, 256)

### `ConcatForwardKernel<T>`
- **Location**: `concat.cu`
- **Primary Function**: Concatenates multiple tensors along specified dimension using pointer array indirection
- **Key Members**:
  - `inputs`: Array of input tensor pointers (const T**)
  - `output`: Output tensor pointer (T*)
  - `offsets`: Cumulative size offsets for input boundaries (const int64_t*)
  - `N`: Product of dimensions before concat dim (int64_t)
  - `D`: Product of dimensions after concat dim (int64_t)
  - `num_inputs`: Number of tensors to concatenate (int64_t)
  - `K_total`: Total size along concat dimension (int64_t)
- **Core Methods**:
  - `ConcatForwardKernel<<<grid, block>>>(inputs, output, offsets, N, D, num_inputs, K_total)`: Uses binary search (UpperBoundI64) to locate source tensor for each output element, O(log num_inputs) per element lookup
  - `ConcatBackwardKernel<<<grid, block>>>(grad_output, grad_inputs, offsets, N, D, num_inputs, K_total)`: Reverse pass dispatching gradients to appropriate input tensors using same indexing strategy
- **Lifecycle**: Device pointer arrays allocated via cudaMallocAsync for each kernel launch, freed after completion

### `Elementwise Operations`
- **Location**: `elementwise.cu`
- **Primary Function**: Template-based elementwise computations supporting unary, binary, and broadcasting operations
- **Key Members**:
  - `CalcOffset(idx, ndim, strides, shape, out_strides)`: Computes linearized memory offsets for broadcasting with stride-based indirection
  - `UnaryForwardKernel<T, Func>`: Applies function object to each element
  - `BinaryForwardKernel<T, Func>`: Applies binary function with automatic broadcasting
- **Core Methods**:
  - `UnaryForward<<<grid, block>>>(output, fn, num_elements, offset, input)`: O(n) element-wise operation with grid-stride loop for large tensors
  - `BinaryForward<<<grid, block>>>(output, fn, ndim, a_strides, a_shape, b_strides, b_shape, out_strides, a, b, num_elements)`: Handles broadcasting via runtime stride computation
  - `BinaryBackwardKernel`: Specialized implementations for float (warp-level reduction) and low-precision types (two-pass histogram strategy for BF16/half)
- **Lifecycle**: Launches with 256-thread blocks, uses dynamic shared memory for broadcast metadata, implements fast atomic operations for BF16/half gradients

### `LayerNormForwardKernel<BLOCK_SIZE, T>`
- **Location**: `layernorm.cu`
- **Primary Function**: Layer normalization with mean and variance computation across feature dimension
- **Key Members**:
  - `input`: Input tensor pointer (const T*)
  - `weight`: Gamma scaling parameter (const T*)
  - `bias`: Beta shifting parameter (const T*)
  - `mean_out`: Optional output for computed mean (float*)
  - `rstd_out`: Optional output for reciprocal standard deviation (float*)
  - `output`: Normalized output (T*)
  - `eps`: Epsilon for numerical stability (float)
  - `embed_dim`: Size of normalization dimension (int)
- **Core Methods**:
  - `LayerNormForwardKernel<<<num_blocks, BLOCK_SIZE>>>(input, weight, bias, mean_out, rstd_out, output, eps, embed_dim)`: Uses CUB BlockReduce for O(n) parallel reduction computing sum and sum of squares in two passes, then normalizes with `y = (x - mean) / sqrt(var + eps) * gamma + beta`
  - `LayerNormBackwardKernel<<<num_blocks, BLOCK_SIZE>>>(input, grad_output, mean, rstd, weight, grad_input, grad_weight, grad_bias, embed_dim, ...)`: Computes gradients using chain rule with optimized block reduction for weight/bias gradients, uses fastAtomicAdd for accumulation
- **Lifecycle**: Each block processes one token (all features), synchronized via shared memory for reduction results

### `MatmulForward` (Linear Layer)
- **Location**: `linear.cu`
- **Primary Function**: Batched matrix multiplication using cuBLAS SGEMM/GEMM routines
- **Key Members**:
  - `input`: Input tensor [*, m, k]
  - `other`: Weight tensor [*, k, n]
  - `output`: Result tensor [*, m, n]
  - `bs`: Batch size (product of leading dimensions)
  - `m, k, n`: Matrix dimensions (int64_t)
- **Core Methods**:
  - `MatmulForward(input, other)`: Calls `cublasGemmStridedBatchedEx` with column-major layout conversion (output.T = other.T * input.T), complexity O(bs * m * n * k)
  - `MatmulBackward(input, other, grad_output)`: Computes two GEMM operations for gradients (grad_input = grad_output * other.T, grad_other = input.T * grad_output), uses type promotion for mixed precision
- **Lifecycle**: Uses persistent cuBLAS handle stored in CudaDevice, supports FP32 and BF16 compute with FP32 accumulation

### `CrossEntropyForwardKernel<BLOCK_SIZE, TargetType, InputType>`
- **Location**: `cross_entropy.cu`
- **Primary Function**: Cross-entropy loss computation with softmax for classification
- **Key Members**:
  - `input_ptr`: Logits [bs, num_classes]
  - `target_ptr`: Class indices [bs]
  - `loss_ptr`: Output loss per sample (InputType*)
  - `bs`: Batch size
  - `num_classes`: Number of classes
- **Core Methods**:
  - `CrossEntropyForwardKernel<<<bs, BLOCK_SIZE>>>(input_ptr, target_ptr, loss_ptr, bs, num_classes)`: Three-pass algorithm: (1) max reduction for stability, (2) sum of exp(max - x), (3) loss = log(sum_exp) - x[target], uses CUB BlockReduce for O(num_classes) parallel reduction per sample
  - `CrossEntropyBackwardKernel<<<bs, BLOCK_SIZE>>>(input_ptr, input_grad_ptr, target_ptr, output_grad_ptr, bs, num_classes)`: Gradient = softmax - one_hot(target), scaled by output gradient
- **Lifecycle**: One block per sample, supports target types uint8 and int64, input types float32/bfloat16

### `GenericReduceKernel<T, ReduceFunc, FinalizeOp, BLOCK_SIZE>`
- **Location**: `reduction.cu`
- **Primary Function**: Generic reduction backend supporting sum, mean, max, min operations
- **Key Members**:
  - `input`: Source tensor (const T*)
  - `output`: Reduced output (T*)
  - `N`: Product of dimensions before reduction axis (int64_t)
  - `H`: Size of reduction axis (int64_t)
  - `W`: Product of dimensions after reduction axis (int64_t)
  - `finalize_op`: Finalization function (MeanFinalize or IdentityFinalize)
- **Core Methods**:
  - `GenericReduceKernel<<<N*W, BLOCK_SIZE>>>(input, output, N, H, W, finalize_op)`: Each block reduces H elements, uses CUB BlockReduce with custom reducer (::cuda::std::plus<>, ::cuda::maximum<>, ::cuda::minimum<>), complexity O(H) per block
  - `GenericReduceBackwardKernel<<<grid, block>>>(grad_input, grad_output, input, reduced, N, H, W, is_mean, is_masked)`: For sum/mean: broadcasts gradient; for max/min: masks gradient to selected indices using equality comparison
- **Lifecycle**: Supports in-place reduction, keeps dimensions or removes based on keep_dim parameter

### `SoftmaxForwardKernel<BLOCK_SIZE, T>`
- **Location**: `softmax.cu`
- **Primary Function**: Softmax activation with normalized exponent computation
- **Key Members**:
  - `outer_size`: Product of dimensions before axis (int64_t)
  - `axis_size`: Size of softmax dimension (int64_t)
  - `inner_size`: Product of dimensions after axis (int64_t)
- **Core Methods**:
  - `SoftmaxForwardKernel<<<dim3(outer_size, inner_size), BLOCK_SIZE>>>(output, input, outer_size, axis_size, inner_size)`: 2D grid where each (outer, inner) point computes softmax over axis, uses max-subtraction for numerical stability: softmax(x) = exp(x - max) / sum(exp(x - max))
  - `SoftmaxBackwardKernel<<<dim3(outer_size, inner_size), BLOCK_SIZE>>>(grad_input, grad_output, output, outer_size, axis_size, inner_size)`: Gradient = output * (grad_output - sum(output * grad_output)), computes dot product reduction
- **Lifecycle**: Block size 256, synchronizes twice per softmax computation (for max and sum)

### `EmbeddingForwardKernel<T>`
- **Location**: `embedding.cu`
- **Primary Function**: Embedding lookup table with sparse gradient updates
- **Key Members**:
  - `input`: Token indices [batch_size, max_seqlen] (const int64_t*)
  - `output`: Embedded vectors [batch_size, max_seqlen, embed_dim] (T*)
  - `weight`: Embedding matrix [vocab_size, embed_dim] (const T*)
  - `vocab_size`: Size of vocabulary (int)
  - `embed_dim`: Embedding dimension (int)
- **Core Methods**:
  - `EmbeddingForwardKernel<<<grid, block>>>(input, output, weight, batch_size, max_seqlen, embed_dim, vocab_size)`: Direct memory copy from weight[token_id] to output position, O(1) per lookup
  - `EmbeddingBackwardKernel<<<grid, block>>>(input_ptr, grad_output_ptr, grad_weight_ptr, num_tokens, embedding_dim, vocab_size)`: Uses atomicAdd to accumulate gradients for duplicate token access, each thread processes one token and loops over embedding_dim
- **Lifecycle**: Forward pass: 256 threads per block; backward: one block per token, requires gradient initialization to zero

### `Transform Operations`
- **Location**: `transform.cu`
- **Primary Function**: Tensor structural transformations without data modification
- **Key Members**:
  - `TrilForwardKernel<T>`: Lower triangular mask (row - col + diagonal >= 0)
  - `TriuForwardKernel<T>`: Upper triangular mask (row - col + diagonal <= 0)
  - `TransposeForwardKernel<T>`: Dimension swapping via stride-based indexing
  - `MaskForwardKernel<T>`: Boolean masking with lead/tail broadcast modes
  - `RepeatInterleaveForwardKernel<T>`: Element repetition along dimension
- **Core Methods**:
  - `TransposeForward(input, dim0, dim1)`: Allocates device memory for strides, computes output coordinates via modulo/division, swaps dim0/dim1 coordinates, computes linear input index, O(1) per element
  - `MaskForward(input, mask, value)`: Detects lead vs tail alignment via `IsLeadMaskShape`/`IsTailMaskShape`, broadcasts mask accordingly, zeros masked positions
  - `TrilForward/TriuForward`: Conditional assignment based on (row - col + diagonal) predicate
- **Lifecycle**: Most operations use cudaMallocAsync for metadata, freed after kernel completion

### `Comm Primitives` (Communication Operations)
- **Location**: `comm.cu`
- **Primary Function**: Distributed training communication operations
- **Key Members**:
  - `Broadcast`: Replicates tensor to multiple devices
  - `Scatter`: Splits tensor across devices along dimension
  - `Gather`: Collects tensors from multiple devices
  - `ReduceAddCoalesced`: Sums gradients from multiple devices to destination
- **Core Methods**:
  - `Broadcast(input_tensors, devices)`: Calls Tensor::To() for each device-device pair, uses device-side copy
  - `ReduceAddCoalesced(grads, destination)`: Initializes output with zeros, iteratively calls AccumulateGrad kernel for each gradient tensor
  - `Scatter(tensor, devices, dim)`: Splits tensor using Tensor::Split, transfers each chunk to target device
  - `Gather(tensors, destination, dim)`: Transfers all tensors to destination, calls StackForward to concatenate
- **Lifecycle**: Pure CPU orchestration with kernel launches for data movement

## 3. API Interface

```cpp
// Gradient Accumulation
void AccumulateGrad(const std::shared_ptr<Tensor>& gradient, float rate,
                   const std::shared_ptr<Tensor>& tensor);
// Accumulates scaled gradient into parameter tensor

void AdamAccumulateGrad(const std::shared_ptr<Tensor>& grad,
                       const std::shared_ptr<Tensor>& param,
                       const std::shared_ptr<Tensor>& m,
                       const std::shared_ptr<Tensor>& v,
                       float learning_rate, float beta1, float beta2,
                       float eps, int64_t t);
// Adam optimizer update with bias correction

// Type Conversion
std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype);
// Converts tensor to specified data type with grid-stride loop

// Tensor Manipulation
std::shared_ptr<Tensor> ConcatForward(const std::vector<std::shared_ptr<Tensor>>& inputs,
                                     int64_t dim);
// Concatenates tensors along dimension using binary search indexing

std::vector<std::shared_ptr<Tensor>> ConcatBackward(
    const std::shared_ptr<Tensor>& grad_output,
    const std::vector<std::vector<int64_t>>& input_dims_list,
    int64_t dim);
// Distributes gradients to input tensors

std::shared_ptr<Tensor> StackForward(const std::vector<std::shared_ptr<Tensor>>& inputs,
                                    int64_t dim);
// Stacks tensors along new dimension

std::vector<std::shared_ptr<Tensor>> SplitForward(const std::shared_ptr<Tensor>& input,
                                                  int64_t split_size, int dim);
// Splits tensor into equal-sized chunks

// Loss Functions
std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor>& input,
                                           const std::shared_ptr<Tensor>& target);
// Computes cross-entropy loss with numerical stability

std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor>& input, int64_t dim);
// Computes softmax activation with max-subtraction

std::shared_ptr<Tensor> LayerNormForward(const std::shared_ptr<Tensor>& input,
                                        const std::shared_ptr<Tensor>& weight,
                                        const std::shared_ptr<Tensor>& bias,
                                        const float eps);
// Returns tuple of (output, mean, rstd)

// Linear Algebra
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor>& input,
                                     const std::shared_ptr<Tensor>& other);
// Batched matrix multiplication via cuBLAS

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor>& input,
                                     const std::shared_ptr<Tensor>& weight,
                                     bool transpose,
                                     const std::shared_ptr<Tensor>& bias);
// Linear transformation: y = x*weight^T + bias

std::shared_ptr<Tensor> OuterForward(const std::shared_ptr<Tensor>& input,
                                    const std::shared_ptr<Tensor>& other);
// Outer product: output[i,j] = input[i] * other[j]

// Elementwise Operations
std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor>& a,
                                  const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor>& a,
                                  const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> DivForward(const std::shared_ptr<Tensor>& a,
                                  const std::shared_ptr<Tensor>& b);
// Arithmetic with broadcasting support

std::shared_ptr<Tensor> ExpForward(const std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> LogForward(const std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor>& input);
std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor>& input);
// Activation functions

// Reduction Operations
std::shared_ptr<Tensor> SumForward(const std::shared_ptr<Tensor>& input,
                                  const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> MeanForward(const std::shared_ptr<Tensor>& input,
                                   const int64_t dim, const bool keep_dim);
std::shared_ptr<Tensor> MaxForward(const std::shared_ptr<Tensor>& input,
                                  const int64_t dim, const bool keep_dim);
// Generic reductions using CUB block reduce

// Indexing and Slicing
std::shared_ptr<Tensor> IndexGatherForward(const std::shared_ptr<Tensor>& input,
                                          const std::shared_ptr<Tensor>& index,
                                          int64_t dim);
// PyTorch-compatible gather: out[i][j][k] = input[index[i][j][k]][j][k]

std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor>& input,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    const std::vector<int64_t>& steps);
// Slices tensor with stride support

std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor>& input,
                                        const std::shared_ptr<Tensor>& weight);
// Embedding lookup: output[b,t,:] = weight[input[b,t],:]

// Transformations
std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor>& input,
                                        int64_t dim0, int64_t dim1);
std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor>& input,
                                    int64_t diagonal);
std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor>& input,
                                    const std::shared_ptr<Tensor>& mask,
                                    float value);
// Structural transformations

// Communication
std::vector<std::shared_ptr<Tensor>> Broadcast(
    const std::vector<std::shared_ptr<Tensor>>& input_tensors,
    const std::vector<const Device*>& devices);
std::vector<std::shared_ptr<Tensor>> ReduceAddCoalesced(
    const std::vector<std::vector<std::shared_ptr<Tensor>>>& grads,
    const Device* destination);
// Distributed operations

// Vocabulary Parallel
std::shared_ptr<Tensor> VocabParallelCrossEntropyBackward(
    const std::shared_ptr<Tensor>& grad_output,
    const std::shared_ptr<Tensor>& softmax_local,
    const std::shared_ptr<Tensor>& target_mask,
    const std::shared_ptr<Tensor>& masked_target,
    const std::shared_ptr<Tensor>& valid_mask_local,
    const int64_t vocab_size_local,
    const int64_t vocab_size_original,
    float label_smoothing);
// Distributed cross-entropy gradient with label smoothing
```

## 4. Usage Example

```cpp
#include "infini_train/include/tensor.h"
#include "infini_train/include/dispatcher.h"

using namespace infini_train;
using namespace infini_train::kernels::cuda;

// Example: Building a Transformer Feed-Forward Network Layer
void TransformerFFNLayer() {
    // Input: [batch_size, seq_len, hidden_dim]
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{32, 128, 768},
        DataType::kBFLOAT16,
        DeviceManager::Instance()->GetDevice(0, DeviceType::kCUDA)
    );

    // Get kernel dispatcher
    auto& dispatcher = Dispatcher::Instance();

    // Step 1: Linear projection (expand to 4x hidden dim)
    // [batch_size*seq_len, hidden_dim] x [hidden_dim, 4*hidden_dim]
    auto kernel_linear1 = dispatcher.GetKernel(
        {DeviceType::kCUDA, "LinearForward"}
    );
    auto weight1 = std::make_shared<Tensor>(
        std::vector<int64_t>{768, 3072},
        DataType::kBFLOAT16,
        input->GetDevice()
    );
    auto bias1 = std::make_shared<Tensor>(
        std::vector<int64_t>{3072},
        DataType::kBFLOAT16,
        input->GetDevice()
    );
    auto hidden = kernel_linear1.Call<std::shared_ptr<Tensor>>(
        input, weight1, false, bias1
    );  // [32, 128, 3072]

    // Step 2: GELU activation (approximated as: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))))
    auto kernel_tanh = dispatcher.GetKernel({DeviceType::kCUDA, "TanhForward"});
    auto x_cubed = dispatcher.GetKernel({DeviceType::kCUDA, "MulForward"})
        .Call<std::shared_ptr<Tensor>>(input, input);
    auto tanh_in = dispatcher.GetKernel({DeviceType::kCUDA, "AddForward"})
        .Call<std::shared_ptr<Tensor>>(input,
            dispatcher.GetKernel({DeviceType::kCUDA, "MulScalarForward"})
                .Call<std::shared_ptr<Tensor>>(x_cubed, 0.044715f)
        );
    auto tanh_out = kernel_tanh.Call<std::shared_ptr<Tensor>>(tanh_in);

    // Step 3: Layer normalization
    auto kernel_layernorm = dispatcher.GetKernel(
        {DeviceType::kCUDA, "LayerNormForward"}
    );
    auto gamma = std::make_shared<Tensor>(
        std::vector<int64_t>{3072},
        DataType::kBFLOAT16,
        input->GetDevice()
    );
    auto beta = std::make_shared<Tensor>(
        std::vector<int64_t>{3072},
        DataType::kBFLOAT16,
        input->GetDevice()
    );
    auto [normed, mean, rstd] = kernel_layernorm
        .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                        std::shared_ptr<Tensor>>>(
            hidden, gamma, beta, 1e-5f
        );

    // Step 4: Output projection (contract back to hidden_dim)
    auto kernel_linear2 = dispatcher.GetKernel(
        {DeviceType::kCUDA, "LinearForward"}
    );
    auto weight2 = std::make_shared<Tensor>(
        std::vector<int64_t>{3072, 768},
        DataType::kBFLOAT16,
        input->GetDevice()
    );
    auto output = kernel_linear2.Call<std::shared_ptr<Tensor>>(
        normed, weight2, true, nullptr
    );  // [32, 128, 768]

    // Backward pass example
    auto grad_output = std::make_shared<Tensor>(
        output->Dims(), DataType::kBFLOAT16, output->GetDevice()
    );

    // Gradient for second linear layer
    auto kernel_linear2_bwd = dispatcher.GetKernel(
        {DeviceType::kCUDA, "LinearBackward"}
    );
    auto [grad_normed, grad_weight2, grad_bias2] = kernel_linear2_bwd
        .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                        std::shared_ptr<Tensor>>>(
            normed, weight2, true, 768, grad_output, false
        );

    // Gradient for layer norm
    auto kernel_layernorm_bwd = dispatcher.GetKernel(
        {DeviceType::kCUDA, "LayerNormBackward"}
    );
    auto [grad_hidden, grad_gamma, grad_beta] = kernel_layernorm_bwd
        .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                        std::shared_ptr<Tensor>>>(
            hidden, gamma, beta, mean, rstd, grad_normed
        );
}
```

## 5. Implementation Details

### Memory Management
- **CUDA Stream Ordering**: All kernels use cudaMallocAsync/cudaFreeAsync for stream-ordered memory allocation, enabling efficient memory reuse without explicit synchronization
- **Gradient Initialization**: Backward kernels require pre-zeroed gradient tensors; most implementations use Fill<T>(0) kernel before accumulation
- **Pointer Arrays**: Operations like Concat/Stack allocate temporary device memory for pointer arrays (T**), freed immediately after kernel completion
- **Type Promotion**: Binary operations and backward passes use WidestType_t trait to determine compute precision (e.g., BF16 + FP32 → FP32)

### Concurrency
- **Block Size Selection**: Most kernels use 256 threads per block for balance between occupancy and shared memory usage; specialized kernels (e.g., Softmax) use 256 for reduction efficiency
- **Grid-Stride Loops**: Elementwise kernels handle tensors larger than grid capacity via for loops with stride `gridDim.x * blockDim.x`
- **CUB Primitives**: LayerNorm and Softmax use cub::BlockReduce for O(log block_size) parallel reduction within each block
- **Atomic Operations**:
  - EmbeddingBackward uses atomicAdd for gradient accumulation (multiple threads may write same embedding row)
  - Elementwise BinaryBackward for BF16/half uses fastAtomicAdd with padded shared memory layout to avoid bank conflicts
  - Implements warp-level reduction for float gradients using __ballot_sync and __shfl_sync

### Performance
- **cuBLAS Integration**: Linear layer uses cublasSgemm (FP32) or cublasGemmEx (BF16) with compute precision CUDA_R_32F for accumulation
- **Numerical Stability**:
  - Softmax/CrossEntropy use max-subtraction: exp(x - max) / sum(exp(x - max))
  - LayerNorm uses rsqrt(var + eps) instead of 1/sqrt for better precision
  - Adam uses bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
- **Low-Precision Optimization**:
  - Elementwise backward for BF16/half uses three strategies based on broadcast pattern:
    1. NoBroadcast: Direct write (no atomics)
    2. TwoPassHist: Shared memory histogram accumulation for small broadcast dimensions (K <= 4096)
    3. BlockReduce: Fallback with padded shared memory (SoA layout) and fast atomics
- **Memory Coalescing**: Kernels favor stride-1 access patterns in innermost dimensions; CalcOffset function computes linearized indices with row-major ordering
- **Vectorization**: Compile-time template specialization enables type-specific optimizations (e.g., __fmadd_rn, __fsqrt_rn intrinsics)

### Error Handling
- **CHECK Macros**: All kernel calls wrapped in CUBLAS_CHECK/CUDA_CHECK for error propagation
- **Shape Validation**: Extensive runtime checks use CHECK_EQ/CHECK_GE/CHECK_LT for dimension compatibility
- **Type Dispatch**: DispatchFunc template with compile-time type lists ensures type safety; unsupported types trigger FATAL log
- **Memory Bounds**: Kernels check `idx < num_elements` to prevent out-of-bounds access

### Dependencies
- **CUB (CUDA Unbound)**: Used for BlockReduce and WarpReduce primitives in LayerNorm, Softmax, CrossEntropy, and Reduction kernels
- **cuBLAS**: Matrix multiplication in linear.cu, outer.cu; uses cublasHandle_t cached in CudaDevice
- **CUDA Runtime**: cudaMallocAsync, cudaFreeAsync, cudaMemcpyAsync for async memory operations
- **GLog**: LOG_LOC macros for error reporting and unsupported data type messages

### Design Patterns
- **Kernel Registration Macro**: Each file uses REGISTER_KERNEL macro to register {DeviceType, kernel_name, function_ptr} in global dispatcher table
- **Functor Pattern**: Elementwise operations use lambda functors captured in template parameters for compile-time optimization
- **CRTP for Dispatch**: DispatchFunc uses visitor pattern on DataTypeList with generic lambda
- **Strategy Pattern for Reduction**: GenericReduceKernel templated on ReduceFunc (::cuda::std::plus<>, ::cuda::maximum<>, ::cuda::minimum<>) and FinalizeOp (MeanFinalize, IdentityFinalize)
- **View Optimization**: NoOpForward creates tensor views with same data pointer but different shape (zero-copy reshape)
