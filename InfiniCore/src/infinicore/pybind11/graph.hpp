#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::graph {
inline void bind(py::module_ &m) {
    // GraphTensorMeta 绑定
    py::class_<GraphTensorMeta>(m, "GraphTensorMeta")
        .def_readonly("shape", &GraphTensorMeta::shape)
        .def_property_readonly("dtype", [](const GraphTensorMeta &meta) {
            return static_cast<int>(meta.dtype);
        })
        .def_readonly("is_input", &GraphTensorMeta::is_input)
        .def("__repr__", [](const GraphTensorMeta &m) {
            std::string shape_str = "(";
            for (size_t i = 0; i < m.shape.size(); ++i) {
                shape_str += std::to_string(m.shape[i]);
                if (i < m.shape.size() - 1) {
                    shape_str += ", ";
                }
            }
            shape_str += ")";
            return "GraphTensorMeta(shape=" + shape_str + ", dtype=" + std::to_string(static_cast<int>(m.dtype)) + ", is_input=" + (m.is_input ? "True" : "False") + ")";
        });

    // GraphOperator 绑定
    py::class_<GraphOperator, std::shared_ptr<GraphOperator>>(m, "GraphOperator")
        .def_property_readonly("op_type", &GraphOperator::op_type)
        .def_property_readonly("tensor_metas", &GraphOperator::tensor_metas)
        .def("__repr__", [](const GraphOperator &op) {
            return "GraphOperator(op_type='" + op.op_type() + "', tensors=" + std::to_string(op.tensor_metas().size()) + ")";
        });

    // Graph 绑定
    py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
        .def(py::init<>())
        .def("run", &Graph::run)
        .def("operators", &Graph::operators, py::return_value_policy::reference_internal)
        .def("__len__", &Graph::size)
        .def("__repr__", [](const Graph &g) {
            return "Graph(operators=" + std::to_string(g.size()) + ")";
        });
}
} // namespace infinicore::graph
