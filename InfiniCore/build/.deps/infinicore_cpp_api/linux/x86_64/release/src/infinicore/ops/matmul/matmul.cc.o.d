{
    files = {
        "src/infinicore/ops/matmul/matmul.cc"
    },
    depfiles_format = "gcc",
    depfiles = "matmul.o: src/infinicore/ops/matmul/matmul.cc  include/infinicore/ops/matmul.hpp include/infinicore/ops/../device.hpp  include/infinicore.h include/infinicore/ops/common/op.hpp  include/infinicore/ops/common/../../context/context.hpp  include/infinicore/ops/common/../../context/../memory.hpp  include/infinicore/ops/common/../../context/../graph/graph.hpp  include/infinicore/ops/common/../../context/../graph/../tensor.hpp  include/infinicore/ops/common/../../context/../graph/../dtype.hpp  include/infiniop.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/ops/add.h  include/infiniop/ops/../operator_descriptor.h  include/infiniop/ops/../handle.h  include/infiniop/ops/../tensor_descriptor.h  include/infiniop/ops/../../infinicore.h  include/infiniop/ops/add_rms_norm.h include/infiniop/ops/attention.h  include/infiniop/ops/gemm.h include/infiniop/ops/swiglu.h  include/infiniop/ops/causal_softmax.h include/infiniop/ops/clip.h  include/infiniop/ops/conv.h include/infiniop/ops/dequantize_awq.h  include/infiniop/ops/gelu.h include/infiniop/ops/gemm.h  include/infiniop/ops/layer_norm.h include/infiniop/ops/logsoftmax.h  include/infiniop/ops/lp_norm.h include/infiniop/ops/mul.h  include/infiniop/ops/ones.h include/infiniop/ops/paged_attention.h  include/infiniop/ops/paged_attention_prefill.h  include/infiniop/ops/paged_caching.h  include/infiniop/ops/random_sample.h include/infiniop/ops/rearrange.h  include/infiniop/ops/relu.h include/infiniop/ops/rms_norm.h  include/infiniop/ops/rope.h include/infiniop/ops/sigmoid.h  include/infiniop/ops/silu.h include/infiniop/ops/softmax.h  include/infiniop/ops/softplus.h include/infiniop/ops/sub.h  include/infiniop/ops/swiglu.h include/infiniop/ops/tanh.h  include/infiniop/ops/topkrouter.h include/infiniop/ops/topksoftmax.h  include/infiniop/ops/zeros.h include/infiniop/tensor_descriptor.h  include/infinirt.h include/infinicore.h  include/infinicore/ops/common/dispatcher.hpp  include/infinicore/ops/gemm.hpp\
",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-Ithird_party/spdlog/include",
            "-I/home/qy/.infini/include",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    }
}