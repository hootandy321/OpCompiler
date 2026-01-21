{
    files = {
        "src/infiniop/ops/topksoftmax/cpu/topksoftmax_cpu.cc"
    },
    depfiles_format = "gcc",
    depfiles = "topksoftmax_cpu.o: src/infiniop/ops/topksoftmax/cpu/topksoftmax_cpu.cc  src/infiniop/ops/topksoftmax/cpu/topksoftmax_cpu.h  src/infiniop/ops/topksoftmax/cpu/../topksoftmax.h  src/infiniop/ops/topksoftmax/cpu/../../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/topksoftmax/cpu/../info.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils.h  include/infinicore.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils/custom_types.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils/rearrange.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils/result.hpp  src/infiniop/ops/topksoftmax/cpu/../../../../utils/check.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils/../utils.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils/infini_status_string.h  src/infiniop/ops/topksoftmax/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils.h  src/infiniop/ops/topksoftmax/cpu/../../../../utils.h  src/infiniop/ops/topksoftmax/cpu/../../../devices/cpu/common_cpu.h  src/infiniop/ops/topksoftmax/cpu/../../../devices/cpu/../../../utils.h  src/infiniop/ops/topksoftmax/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/topksoftmax/cpu/../../../devices/cpu/../../handle.h  include/infiniop/handle.h  src/infiniop/ops/topksoftmax/cpu/../../../reduce/cpu/reduce.h  src/infiniop/ops/topksoftmax/cpu/../../../reduce/cpu/../../../utils.h\
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