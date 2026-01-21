{
    files = {
        "src/infiniop/ops/tanh/cpu/tanh_cpu.cc"
    },
    depfiles_format = "gcc",
    depfiles = "tanh_cpu.o: src/infiniop/ops/tanh/cpu/tanh_cpu.cc  src/infiniop/ops/tanh/cpu/tanh_cpu.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/elementwise_cpu.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/common_cpu.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils.h  include/infinicore.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/custom_types.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/rearrange.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/result.hpp  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/check.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/../utils.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/infini_status_string.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/cpu_handle.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../devices/cpu/../../handle.h  include/infiniop/handle.h include/infiniop/../infinicore.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../elementwise.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/tanh/cpu/../../../elementwise/cpu/../../../utils.h\
",
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
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-fopenmp",
            "-DNDEBUG"
        }
    }
}