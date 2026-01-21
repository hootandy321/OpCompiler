# `Graph` Computation Graph Execution Framework Core Implementation Documentation

This module implements a computation graph execution framework for InfiniCore, providing a declarative API for building and executing computational operator graphs with support for multiple hardware backends through a dispatch-based architecture. The framework separates graph construction (recording mode) from graph execution, enabling optimization opportunities and efficient hardware abstraction.

## 1. Module Structure

- **`graph.hpp`**: Core computation graph framework defining graph containers, operator abstraction, tensor wrappers, and macro-based operator registration/dispatch system

## 2. Core Classes

### `GraphTensor`
- **Location**: `graph.hpp:12-15`
- **Primary Function**: Tensor wrapper class for graph-mode operations that inherits from the base `Tensor` class, enabling tensors to participate in computation graph construction and execution
- **Key Members**:
  - (Implicit members from `Tensor` base class)
- **Core Methods**:
  - `GraphTensor(const Tensor &)`: Copy constructor that wraps an existing tensor into graph-compatible form
- **Lifecycle**: Value semantic wrapper, constructed from existing Tensor instances, follows standard C++ object lifecycle

### `GraphOperator`
- **Location**: `graph.hpp:17-29`
- **Primary Function**: Abstract base class for all computational operators in the graph, encapsulating hardware-specific execution logic through function pointers to plan, run, and cleanup routines
- **Key Members**:
  - `planned_meta_`: `void*` pointer to hardware-specific execution metadata (kernel configurations, memory layouts, etc.)
  - `runner_`: `run_schema` function pointer (`void (*)(void*)`) to hardware-specific execution routine
  - `deleter_`: `cleanup_schema` function pointer (`void (*)(void**)`) for resource cleanup
- **Core Methods**:
  - `run() const`: Executes the operator using the stored runner function pointer and planned metadata
  - `~GraphOperator()`: Destructor that invokes cleanup schema to release hardware resources
- **Lifecycle**: Managed through `std::shared_ptr`, constructed by operator factories, destroyed when graph is destroyed

### `Graph`
- **Location**: `graph.hpp:31-44`
- **Primary Function**: Computation graph container that holds a sequence of operators and provides batch execution of the entire computational graph
- **Key Members**:
  - `op_list_`: `std::vector<std::shared_ptr<GraphOperator>>` maintaining ordered sequence of operators
- **Core Methods**:
  - `run() const`: Executes all operators in the graph sequentially in insertion order
  - `add_operator(std::shared_ptr<GraphOperator> op)`: Protected method for appending operators to the graph
- **Lifecycle**: Default constructible, manages operator lifetimes through shared pointers, friendship with `GraphManager` allows external manipulation

## 3. API Interface

```cpp
// Core execution interface
void run() const;
// Executes the computation graph or individual operator

void add_operator(std::shared_ptr<GraphOperator> op);
// Appends operator to graph (protected, called by GraphManager)

// Macro-based operator definition interface
#define INFINICORE_GRAPH_OP_CLASS(__OP_NAME__, ...)
// Defines operator class inheriting from GraphOperator with typed schema

#define INFINICORE_GRAPH_OP_DISPATCH(__DEVICE_TYPE__, ...)
// Resolves hardware-specific plan/run/cleanup functions for operator

#define INFINICORE_GRAPH_OP_RECORD_OR_RUN(__OP_NAME__, ...)
// Conditionally records operator to graph or executes immediately based on context mode

#define INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(__OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)
// Registers operator implementations across all device types using static initialization
```

## 4. Usage Example

```cpp
// Example: Defining and using a custom MatMul operator in the graph framework

// 1. Define operator class using macro
INFINICORE_GRAPH_OP_CLASS(MatMulOp, const Tensor &, const Tensor &, Tensor)

// 2. Implement operator registration
MatMulOp::MatMulOp(const Tensor &A, const Tensor &B, Tensor &C) {
    INFINICORE_GRAPH_OP_DISPATCH(DeviceType::CUDA, A, B, C);
}

void MatMulOp::execute(const Tensor &A, const Tensor &B, Tensor &C) {
    // Hardware-specific implementation selected by dispatcher
}

// 3. Register implementations for all backends
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    MatMulOp,
    MatMulPlanCUDA,   // Planning function
    MatMulRunCUDA,    // Execution function
    MatMulCleanupCUDA // Cleanup function
)

// 4. Use operator in user code
void matmul_example() {
    Tensor A = /* create tensor A */;
    Tensor B = /* create tensor B */;
    Tensor C = /* create output tensor C */;

    // Automatically records to graph or executes based on context
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MatMulOp, A, B, C);

    // When recording mode is active, operator is added to current graph
    // When execution mode is active, operator runs immediately
}
```

## 5. Implementation Details

- **Memory Management**: Reference-counted operator lifecycle through `std::shared_ptr`, ensuring operators remain valid during graph execution and are automatically destroyed when graph is destroyed. Hardware-specific metadata (`planned_meta_`) managed through opaque `void*` pointer with custom cleanup function

- **Hardware Abstraction**: Three-phase execution model separating planning (kernel selection, memory layout), execution (actual computation), and cleanup (resource deallocation). Function pointers stored per-operator instance enable zero-overhead virtual dispatch without C++ vtable

- **Performance**: Macro-based registration system uses static initialization for dispatcher registration, eliminating runtime registration overhead. Typed schema (`using schema = void (*)(Args...)`) provides compile-time type safety while maintaining runtime hardware dispatch flexibility

- **Error Handling**: Relies on hardware-specific implementations for error checking and propagation. Framework assumes registered implementations are valid; crashes occur if dispatchers return null function pointers

- **Dependencies**: Depends on `infinicore/Tensor` base class, `infinicore/context` for recording mode detection (referenced but not included), and `common::OpDispatcher<T>` template for device-specific dispatch resolution

- **Design Patterns**:
  - **Strategy Pattern**: Hardware-specific implementations selected at runtime through dispatchers
  - **Builder Pattern**: Graph constructed incrementally through operator recording
  - **RAII**: Resource cleanup handled through operator destructors and cleanup schemas
  - **Static Registration**: Compile-time operator registration using function-local static variables and lambda-based initialization

- **Macro Metaprogramming**: Complex macro system (`INFINICORE_GRAPH_OP_CLASS`, `INFINICORE_GRAPH_OP_DISPATCH`, `INFINICORE_GRAPH_OP_RECORD_OR_RUN`) generates boilerplate code for operator definition, dispatch resolution, and conditional execution/recording, maintaining DRY principle while providing type-safe interfaces
