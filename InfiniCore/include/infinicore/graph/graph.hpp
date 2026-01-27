#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "../tensor.hpp"

namespace infinicore::graph {
// Forward declarations
class GraphManager;

/**
 * 张量元信息 - 用于记录 Graph 中每个张量的形状和类型
 */
struct GraphTensorMeta {
    Shape shape;
    DataType dtype;
    bool is_input; // true=输入张量, false=输出张量

    GraphTensorMeta(const Shape &s, DataType d, bool input)
        : shape(s), dtype(d), is_input(input) {}
};

class GraphTensor : public Tensor {
public:
    GraphTensor(const Tensor &);
};

class GraphOperator {

public:
    // 获取算子元信息
    const std::string &op_type() const { return op_type_; }
    const std::vector<GraphTensorMeta> &tensor_metas() const { return tensor_metas_; }

    void run() const;
    ~GraphOperator();

protected:
    // 设置算子类型（由宏调用）
    void set_op_type(const char *name) { op_type_ = name; }

    // 变参模板：自动从参数中提取 Tensor 的 shape/dtype
    template <typename... Args>
    void capture_tensors(Args &&...args) {
        size_t idx = 0;
        _capture_impl(idx, std::forward<Args>(args)...);
    }

private:
    // 终止递归
    void _capture_impl(size_t &) {}

    // 递归提取参数
    template <typename First, typename... Rest>
    void _capture_impl(size_t &idx, First &&first, Rest &&...rest) {
        _try_capture_tensor(idx, std::forward<First>(first));
        _capture_impl(idx, std::forward<Rest>(rest)...);
    }

    // SFINAE: 捕获 Tensor 类型参数
    template <typename T>
    auto _try_capture_tensor(size_t &idx, T &&t)
        -> std::enable_if_t<std::is_same_v<std::decay_t<T>, Tensor>> {
        // 约定：第一个 Tensor 是输出，其余是输入
        bool is_input = (idx > 0);
        tensor_metas_.emplace_back(t->shape(), t->dtype(), is_input);
        ++idx;
    }

    // SFINAE: 非 Tensor 类型忽略
    template <typename T>
    auto _try_capture_tensor(size_t &, T &&)
        -> std::enable_if_t<!std::is_same_v<std::decay_t<T>, Tensor>> {}

protected:
    std::string op_type_;
    std::vector<GraphTensorMeta> tensor_metas_;

    using run_schema = void (*)(void *);
    using cleanup_schema = void (*)(void **);
    void *planned_meta_;
    run_schema runner_;
    cleanup_schema deleter_;
};

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    void run() const;

    // 获取算子列表
    const std::vector<std::shared_ptr<GraphOperator>> &operators() const {
        return op_list_;
    }

    size_t size() const { return op_list_.size(); }

protected:
    void add_operator(std::shared_ptr<GraphOperator> op);

    std::vector<std::shared_ptr<GraphOperator>> op_list_;

    friend class GraphManager;
};
} // namespace infinicore::graph

#define INFINICORE_GRAPH_OP_CLASS(__OP_NAME__, ...)                        \
    class __OP_NAME__ : public graph::GraphOperator {                      \
    public:                                                                \
        using schema = void (*)(__VA_ARGS__);                              \
        using plan_schema = void *(*)(__VA_ARGS__);                        \
        static common::OpDispatcher<plan_schema> &plan_dispatcher();       \
        static common::OpDispatcher<run_schema> &run_dispatcher();         \
        static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher(); \
        __OP_NAME__(__VA_ARGS__);                                          \
        static void execute(__VA_ARGS__);                                  \
    };

#define INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(__OP_NAME__)                                  \
    common::OpDispatcher<__OP_NAME__::plan_schema> &__OP_NAME__::plan_dispatcher() {       \
        static common::OpDispatcher<__OP_NAME__::plan_schema> dispatcher_;                 \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::run_schema> &__OP_NAME__::run_dispatcher() {         \
        static common::OpDispatcher<__OP_NAME__::run_schema> dispatcher_;                  \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::cleanup_schema> &__OP_NAME__::cleanup_dispatcher() { \
        static common::OpDispatcher<__OP_NAME__::cleanup_schema> dispatcher_;              \
        return dispatcher_;                                                                \
    }

#define INFINICORE_GRAPH_OP_DISPATCH(__DEVICE_TYPE__, ...)                  \
    planned_meta_ = plan_dispatcher().lookup(__DEVICE_TYPE__)(__VA_ARGS__); \
    runner_ = run_dispatcher().lookup(__DEVICE_TYPE__);                     \
    deleter_ = cleanup_dispatcher().lookup(__DEVICE_TYPE__);

#define INFINICORE_GRAPH_OP_RECORD_OR_RUN(__OP_NAME__, ...) \
    auto op = std::make_shared<__OP_NAME__>(__VA_ARGS__);   \
    op->set_op_type(#__OP_NAME__);                          \
    op->capture_tensors(__VA_ARGS__);                       \
    if (context::isGraphRecording()) {                      \
        context::addGraphOperator(op);                      \
    } else {                                                \
        op->run();                                          \
    }

#define INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(__OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__) \
    static bool registered = []() {                                                               \
        __OP_NAME__::plan_dispatcher().registerAll(__PLAN_F__, false);                            \
        __OP_NAME__::run_dispatcher().registerAll(__RUN_F__, false);                              \
        __OP_NAME__::cleanup_dispatcher().registerAll(__CLEANUP_F__, false);                      \
        return true;                                                                              \
    }();
