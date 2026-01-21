{
    files = {
        "src/infinicore/graph/graph.cc"
    },
    depfiles_format = "gcc",
    depfiles = "graph.o: src/infinicore/graph/graph.cc  src/infinicore/graph/graph_manager.hpp  include/infinicore/graph/graph.hpp  include/infinicore/graph/../tensor.hpp  include/infinicore/graph/../device.hpp include/infinicore.h  include/infinicore/graph/../dtype.hpp  include/infinicore/graph/../memory.hpp include/infiniop.h  include/infiniop/handle.h include/infiniop/../infinicore.h  include/infiniop/ops/add.h include/infiniop/ops/../operator_descriptor.h  include/infiniop/ops/../handle.h  include/infiniop/ops/../tensor_descriptor.h  include/infiniop/ops/../../infinicore.h  include/infiniop/ops/add_rms_norm.h include/infiniop/ops/attention.h  include/infiniop/ops/gemm.h include/infiniop/ops/swiglu.h  include/infiniop/ops/causal_softmax.h include/infiniop/ops/clip.h  include/infiniop/ops/conv.h include/infiniop/ops/dequantize_awq.h  include/infiniop/ops/gelu.h include/infiniop/ops/gemm.h  include/infiniop/ops/layer_norm.h include/infiniop/ops/logsoftmax.h  include/infiniop/ops/lp_norm.h include/infiniop/ops/mul.h  include/infiniop/ops/ones.h include/infiniop/ops/paged_attention.h  include/infiniop/ops/paged_attention_prefill.h  include/infiniop/ops/paged_caching.h  include/infiniop/ops/random_sample.h include/infiniop/ops/rearrange.h  include/infiniop/ops/relu.h include/infiniop/ops/rms_norm.h  include/infiniop/ops/rope.h include/infiniop/ops/sigmoid.h  include/infiniop/ops/silu.h include/infiniop/ops/softmax.h  include/infiniop/ops/softplus.h include/infiniop/ops/sub.h  include/infiniop/ops/swiglu.h include/infiniop/ops/tanh.h  include/infiniop/ops/topkrouter.h include/infiniop/ops/topksoftmax.h  include/infiniop/ops/zeros.h include/infiniop/tensor_descriptor.h  src/infinicore/graph/../utils.hpp  src/infinicore/graph/../../utils/infini_status_string.h  third_party/spdlog/include/spdlog/cfg/env.h  third_party/spdlog/include/spdlog/cfg/helpers.h  third_party/spdlog/include/spdlog/common.h  third_party/spdlog/include/spdlog/details/null_mutex.h  third_party/spdlog/include/spdlog/tweakme.h  third_party/spdlog/include/spdlog/fmt/fmt.h  third_party/spdlog/include/spdlog/fmt/bundled/format.h  third_party/spdlog/include/spdlog/fmt/bundled/base.h  third_party/spdlog/include/spdlog/fmt/bundled/format.h  third_party/spdlog/include/spdlog/fmt/bundled/format-inl.h  third_party/spdlog/include/spdlog/common-inl.h  third_party/spdlog/include/spdlog/cfg/helpers-inl.h  third_party/spdlog/include/spdlog/details/os.h  third_party/spdlog/include/spdlog/details/os-inl.h  third_party/spdlog/include/spdlog/details/registry.h  third_party/spdlog/include/spdlog/details/periodic_worker.h  third_party/spdlog/include/spdlog/details/periodic_worker-inl.h  third_party/spdlog/include/spdlog/details/registry-inl.h  third_party/spdlog/include/spdlog/logger.h  third_party/spdlog/include/spdlog/details/backtracer.h  third_party/spdlog/include/spdlog/details/circular_q.h  third_party/spdlog/include/spdlog/details/log_msg_buffer.h  third_party/spdlog/include/spdlog/details/log_msg.h  third_party/spdlog/include/spdlog/details/log_msg-inl.h  third_party/spdlog/include/spdlog/details/log_msg_buffer-inl.h  third_party/spdlog/include/spdlog/details/backtracer-inl.h  third_party/spdlog/include/spdlog/logger-inl.h  third_party/spdlog/include/spdlog/pattern_formatter.h  third_party/spdlog/include/spdlog/formatter.h  third_party/spdlog/include/spdlog/pattern_formatter-inl.h  third_party/spdlog/include/spdlog/details/fmt_helper.h  third_party/spdlog/include/spdlog/mdc.h  third_party/spdlog/include/spdlog/sinks/sink.h  third_party/spdlog/include/spdlog/sinks/sink-inl.h  third_party/spdlog/include/spdlog/sinks/ansicolor_sink.h  third_party/spdlog/include/spdlog/details/console_globals.h  third_party/spdlog/include/spdlog/sinks/ansicolor_sink-inl.h  third_party/spdlog/include/spdlog/spdlog.h  third_party/spdlog/include/spdlog/details/synchronous_factory.h  third_party/spdlog/include/spdlog/version.h  third_party/spdlog/include/spdlog/spdlog-inl.h\
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