#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class CausalSoftmax {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor causal_softmax(Tensor input);
void causal_softmax_(Tensor output, Tensor input);
} // namespace infinicore::op
