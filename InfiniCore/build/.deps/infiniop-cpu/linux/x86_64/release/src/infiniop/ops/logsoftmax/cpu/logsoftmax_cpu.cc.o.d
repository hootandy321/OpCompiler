{
    files = {
        "src/infiniop/ops/logsoftmax/cpu/logsoftmax_cpu.cc"
    },
    depfiles_format = "gcc",
    depfiles = "logsoftmax_cpu.o: src/infiniop/ops/logsoftmax/cpu/logsoftmax_cpu.cc  src/infiniop/ops/logsoftmax/cpu/logsoftmax_cpu.h  src/infiniop/ops/logsoftmax/cpu/../logsoftmax.h  src/infiniop/ops/logsoftmax/cpu/../../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/logsoftmax/cpu/../info.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils.h include/infinicore.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/custom_types.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/rearrange.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/result.hpp  src/infiniop/ops/logsoftmax/cpu/../../../../utils/check.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/../utils.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/infini_status_string.h  src/infiniop/ops/logsoftmax/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils.h  src/infiniop/ops/logsoftmax/cpu/../../../devices/cpu/common_cpu.h  src/infiniop/ops/logsoftmax/cpu/../../../devices/cpu/../../../utils.h  src/infiniop/ops/logsoftmax/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/logsoftmax/cpu/../../../devices/cpu/../../handle.h  include/infiniop/handle.h  src/infiniop/ops/logsoftmax/cpu/../../../reduce/cpu/reduce.h  src/infiniop/ops/logsoftmax/cpu/../../../reduce/cpu/../../../utils.h\
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