{
    files = {
        "src/infiniop-test/src/ops/ones.cpp"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Werror",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-Ithird_party/spdlog/include",
            "-I/home/qy/.infini/include",
            "-I/home/qy/src/Infini/InfiniCore/src/infiniop-test/include",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fopenmp",
            "-DNDEBUG"
        }
    },
    depfiles_format = "gcc",
    depfiles = "ones.o: src/infiniop-test/src/ops/ones.cpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/ops.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/test.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/gguf.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/file_mapping.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/tensor.hpp  include/infiniop.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/ops/add.h  include/infiniop/ops/../operator_descriptor.h  include/infiniop/ops/../handle.h  include/infiniop/ops/../tensor_descriptor.h  include/infiniop/ops/../../infinicore.h  include/infiniop/ops/add_rms_norm.h include/infiniop/ops/attention.h  include/infiniop/ops/gemm.h include/infiniop/ops/swiglu.h  include/infiniop/ops/causal_softmax.h include/infiniop/ops/clip.h  include/infiniop/ops/conv.h include/infiniop/ops/dequantize_awq.h  include/infiniop/ops/gelu.h include/infiniop/ops/gemm.h  include/infiniop/ops/layer_norm.h include/infiniop/ops/logsoftmax.h  include/infiniop/ops/lp_norm.h include/infiniop/ops/mul.h  include/infiniop/ops/ones.h include/infiniop/ops/paged_attention.h  include/infiniop/ops/paged_attention_prefill.h  include/infiniop/ops/paged_caching.h  include/infiniop/ops/random_sample.h include/infiniop/ops/rearrange.h  include/infiniop/ops/relu.h include/infiniop/ops/rms_norm.h  include/infiniop/ops/rope.h include/infiniop/ops/sigmoid.h  include/infiniop/ops/silu.h include/infiniop/ops/softmax.h  include/infiniop/ops/softplus.h include/infiniop/ops/sub.h  include/infiniop/ops/swiglu.h include/infiniop/ops/tanh.h  include/infiniop/ops/topkrouter.h include/infiniop/ops/topksoftmax.h  include/infiniop/ops/zeros.h include/infiniop/tensor_descriptor.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/utils.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils.h  include/infinicore.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/custom_types.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/rearrange.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/result.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/check.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/../utils.h  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/../../utils/infini_status_string.h  include/infinirt.h include/infinicore.h\
"
}