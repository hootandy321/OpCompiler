# InfiniLM Engine Python Bindings Core Implementation Documentation

This module provides Python bindings for the InfiniLM distributed inference engine using pybind11. It acts as the bridge layer between Python frontend and C++ backend, exposing the InferEngine, distributed configuration, and worker management APIs to Python while managing tensor parallelism across multiple devices.

## 1. Module Structure

- **`engine.hpp`**: Pybind11 binding definitions for InferEngine, DistConfig, Input/Output structs, and their complete Python API surface

## 2. Core Classes

### `DistConfig`
- **Location**: `engine.hpp` (binds `/home/qy/src/Infini/InfiniLM/csrc/engine/distributed/dist_config.hpp`)
- **Primary Function**: Configuration structure for tensor parallelism distribution across multiple devices
- **Key Members**:
  - `tp_device_ids`: `std::vector<int>` - Device IDs assigned to each rank in tensor parallelism group
- **Constructors**:
  - `DistConfig()`: Default constructor, empty device list (single device mode)
  - `DistConfig(int tp_size)`: Constructor with tensor parallel size, auto-assigns device IDs from 0 to tp_size-1
  - `DistConfig(const std::vector<int>& tp_device_ids)`: Constructor with explicit device ID list
- **String Conversion**: Implements `operator std::string()` for human-readable representation
- **Python Binding Features**:
  - Read/write access to `tp_device_ids` field
  - `__repr__` and `__str__` methods for debugging
  - Multiple constructor overloads exposed with default arguments

### `InferEngine`
- **Location**: `engine.hpp` (binds `/home/qy/src/Infini/InfiniLM/csrc/engine/infer_engine.hpp`)
- **Primary Function**: Main inference orchestrator managing multiple RankWorker instances for distributed model execution with KV cache support
- **Key Members**:
  - `workers_`: `std::vector<std::unique_ptr<RankWorker>>` - Worker threads, one per tensor parallel rank
  - `communication_group_`: `CommunicationGroup` - Manages device communicators (infiniccl) for collective operations
  - `model_config_`: `const InfinilmModel::Config&` - Model architecture configuration reference
  - `cache_config_`: `std::unique_ptr<cache::CacheConfig>` - KV cache configuration (polymorphic type)
- **Core Methods**:
  - `InferEngine(config, dist_config, device_type, cache_config)`: Constructor initializing workers, communication group, and cache system. Accepts shared_ptr to CacheConfig for Python interop
  - `load_param(name, param)`: Distributes parameter tensor to all workers; each worker extracts its tensor parallel shard independently
  - `state_dict()`: Returns vector of parameter dictionaries (one per rank), mapping parameter names to Tensor objects
  - `forward(input)`: Executes synchronized inference across all ranks, returns Output from rank 0
  - `reset_cache(cache_config)`: Reconfigures KV cache with new CacheConfig (supports runtime cache resize/type change)
  - `get_cache_config()`: Returns shared_ptr to CacheConfig with ownership transfer (uses unique_copy() for polymorphic types)
  - `get_dist_config()`: Const accessor for distributed configuration
- **Lifecycle**: Shared ownership via `std::shared_ptr<InferEngine>` required by pybind11 for proper Python object lifetime management
- **Python Binding Features**:
  - Factory-based constructor using lambda with py::none() default for optional cache_config
  - Automatic handling of CacheConfig nullptr when py::none() passed
  - Default device_type uses InfiniCore context device
  - `__repr__` exposes distributed configuration string

### `InferEngine::Input`
- **Location**: `engine.hpp` (binds RankWorker::Input from `/home/qy/src/Infini/InfiniLM/csrc/engine/rank_worker.hpp`)
- **Primary Function**: Encapsulates all tensors and sampling parameters for a single inference request
- **Key Members**:
  - `input_ids`: `std::optional<Tensor>` - Token IDs, shape `[batch, seq_len]`
  - `position_ids`: `std::optional<Tensor>` - Position embeddings, shape `[batch, seq_len]` or `[seq_len]`
  - `cache_lengths`: `std::optional<Tensor>` - Cached sequence lengths per request, shape `[num_requests]`
  - `input_lengths`: `std::optional<Tensor>` - Request lengths in continuous batching, shape `[num_requests]`
  - `input_offsets`: `std::optional<Tensor>` - Request offsets in continuous batching, shape `[num_requests]`
  - `block_tables`: `std::optional<Tensor>` - Paged attention block IDs, shape `[batch, max_block_table_length]`
  - `slot_mapping`: `std::optional<Tensor>` - Paged attention slot IDs, shape `[seq]`
  - `temperature`: `float` (default 1.0) - Sampling temperature
  - `top_k`: `int` (default 50) - Top-k sampling parameter
  - `top_p`: `float` (default 1.0) - Nucleus sampling parameter
  - `random_val`: `float` (default 0.1) - Random seed value
- **Conversion**: `to_model_input()` method converts to InfinilmModel::Input for model forward pass
- **Python Binding Features**:
  - Flexible constructor accepting all tensors as optional keyword arguments
  - Additional sampling params (temperature, top_k, top_p) extracted from py::kwargs using contains() and cast()
  - All fields exposed with def_readwrite for direct Python access
  - Smart use of std::optional with std::nullopt defaults

### `InferEngine::Output`
- **Location**: `engine.hpp` (binds RankWorker::Output)
- **Primary Function**: Encapsulates inference results
- **Key Members**:
  - `output_ids`: `Tensor` - Generated token IDs tensor
- **Python Binding Features**:
  - Simple data class with single exported field
  - Docstring annotation for output_ids

## 3. API Interface

```cpp
// Distributed Configuration
namespace infinilm::engine::distributed {
    struct DistConfig {
        std::vector<int> tp_device_ids;  // Device IDs for tensor parallelism

        DistConfig();  // Empty = single device
        explicit DistConfig(int tp_size);  // Auto-assign 0..tp_size-1
        explicit DistConfig(const std::vector<int>& tp_device_ids);

        explicit operator std::string() const;  // Human-readable representation
    };
}

// Main Inference Engine
namespace infinilm::engine {
    class InferEngine {
    public:
        using Input = RankWorker::Input;
        using Output = RankWorker::Output;

        // Constructor with Python-compatible optional cache_config
        InferEngine(
            const InfinilmModel::Config &config,
            const distributed::DistConfig &distributed_config = distributed::DistConfig(),
            infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
            const cache::CacheConfig *cache_config = nullptr  // nullptr = use default
        );

        // Parameter Management
        void load_param(const std::string &name, const infinicore::Tensor &param);
        std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

        // Inference
        Output forward(const Input &input);

        // Cache Management
        void reset_cache(const cache::CacheConfig *new_config);
        const cache::CacheConfig *get_cache_config() const;

        // Distributed Info
        const distributed::DistConfig &get_dist_config() const;

        ~InferEngine();
    };

    // Input Structure with Sampling Parameters
    struct RankWorker::Input {
        std::optional<infinicore::Tensor> input_ids;        // [batch, seq_len]
        std::optional<infinicore::Tensor> position_ids;     // [batch, seq_len] or [seq_len]
        std::optional<infinicore::Tensor> cache_lengths;    // [num_requests]
        std::optional<infinicore::Tensor> input_lengths;    // [num_requests]
        std::optional<infinicore::Tensor> input_offsets;    // [num_requests]
        std::optional<infinicore::Tensor> block_tables;     // [batch, max_blocks]
        std::optional<infinicore::Tensor> slot_mapping;     // [seq]

        float temperature{1.0};   // Sampling temperature
        int top_k{50};            // Top-k sampling
        float top_p{1.0};         // Nucleus sampling
        float random_val{0.1};    // Random seed

        InfinilmModel::Input to_model_input() const;
    };

    // Output Structure
    struct RankWorker::Output {
        infinicore::Tensor output_ids;  // Generated tokens
    };
}
```

## 4. Usage Example

```python
import infinilm
from infinicore import Device, Tensor

# Configure tensor parallelism across 4 GPUs
dist_config = infinilm.engine.DistConfig(tp_size=4)
# Or specify device IDs explicitly:
# dist_config = infinilm.engine.DistConfig(tp_device_ids=[0, 2, 4, 6])

# Configure KV cache (e.g., paged attention cache)
cache_config = infinilm.cache.PagedCacheConfig(
    num_blocks=1024,
    block_size=16,
    dtype=infinicore.DType.float16
)

# Model configuration (model-specific subclass)
model_config = infinilm.models.LlamaConfig(
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    # ... other model params
)

# Create distributed inference engine
engine = infinilm.engine.InferEngine(
    config=model_config,
    distributed_config=dist_config,
    device_type=Device.Type.CUDA,
    cache_config=cache_config
)

# Load model parameters (distributes across all 4 workers)
for name, param in pretrained_weights.items():
    engine.load_param(name, param)

# Prepare batch input
input_ids = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # Shape: [2, 4]
position_ids = Tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

# Configure sampling
input_config = infinilm.engine.InferEngine.Input(
    input_ids=input_ids,
    position_ids=position_ids,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

# Run distributed inference (synchronized across all 4 workers)
output = engine.forward(input_config)
print(output.output_ids)  # Generated token IDs

# Access parameters from all ranks
all_params = engine.state_dict()  # List of 4 dicts (one per rank)
rank_0_params = all_params[0]

# Reconfigure cache at runtime
new_cache_config = infinilm.cache.PagedCacheConfig(num_blocks=2048, block_size=16)
engine.reset_cache(new_cache_config)

# Get current cache configuration
current_cache = engine.get_cache_config()
print(current_cache.num_blocks)  # 2048
```

## 5. Implementation Details

**Python Interoperability Strategy**:
- Uses pybind11's `std::shared_ptr` support for automatic reference counting between C++ and Python
- Factory-based constructors with lambda functions enable complex default argument handling (e.g., converting py::none() to nullptr)
- Explicit ownership transfer via `unique_copy()` for polymorphic CacheConfig types
- `def_readwrite` provides direct Python attribute access without getter/setter boilerplate

**Distributed Execution Model**:
- **Tensor Parallelism**: Model weights sharded across `tp_size` workers, each worker assigned a specific GPU via `tp_device_ids`
- **Communication**: InfiniCCL communicators created per rank in `CommunicationGroup`, enabling collective operations (all-reduce, all-gather)
- **Synchronization**: `forward()` blocks until all workers complete, returns output from rank 0
- **Parameter Loading**: Each worker receives full parameter, extracts its shard via RankWorker's internal logic

**Memory and Resource Management**:
- **Worker Lifecycle**: Each RankWorker owns a dedicated std::thread executing a command loop (INIT -> LOAD -> RUN -> RESET_CACHE -> STOP)
- **Cache Architecture**: Polymorphic `CacheConfig` enables runtime selection between paged, contiguous, or other KV cache strategies
- **Thread Safety**: RankWorker uses std::mutex + std::condition_variable for job queue synchronization
- **Device Binding**: Each rank binds to its assigned GPU device before worker thread starts

**Continuous Batching Support**:
- Input structure supports variable-length sequences via `input_lengths` and `input_offsets` tensors
- Paged attention enabled through `block_tables` (mapping requests to memory blocks) and `slot_mapping` (mapping tokens to physical cache slots)
- `cache_lengths` tracks per-request prefix length for autoregressive generation

**Error Handling and Validation**:
- Constructor validates DistConfig contains valid device IDs
- CacheConfig nullptr handling in reset_cache() (reset to default)
- State dict returns copy of parameters to prevent external modification

**Design Patterns**:
- **Factory Pattern**: Lambda-based constructors in pybind11 enable complex initialization logic
- **Command Pattern**: RankWorker uses Command enum for job dispatch (INIT, LOAD, RUN, etc.)
- **Producer-Consumer**: Each worker thread implements producer-consumer with condition variable sync
- **Strategy Pattern**: Polymorphic CacheConfig allows pluggable cache backends

**Performance Considerations**:
- All-to-all communication minimized via tensor parallel sharding
- Worker threads enable concurrent device execution (GPU kernels run in parallel)
- Cache reset reuses existing allocations when possible (size-dependent)
- Optional tensors reduce memory overhead for unused features (e.g., block_tables only needed for paged cache)
