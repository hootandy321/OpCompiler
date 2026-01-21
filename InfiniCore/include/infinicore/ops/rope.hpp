#pragma once

#include "../device.hpp"
#include "../nn/rope.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class RoPE {
public:
    using schema = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, infinicore::nn::RoPE::Algo);
    static void execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_cache, infinicore::nn::RoPE::Algo algo);
    static common::OpDispatcher<schema> &dispatcher();
};

// Internal function
void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo);

// Public API that uses infinicore::nn::RoPE::Algo
Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo);
} // namespace infinicore::op
