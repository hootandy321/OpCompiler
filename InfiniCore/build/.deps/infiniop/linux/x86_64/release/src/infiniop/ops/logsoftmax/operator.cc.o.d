{
    files = {
        "src/infiniop/ops/logsoftmax/operator.cc"
    },
    depfiles_format = "gcc",
    depfiles = "operator.o: src/infiniop/ops/logsoftmax/operator.cc  src/infiniop/ops/logsoftmax/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/logsoftmax/../../handle.h include/infiniop/handle.h  include/infiniop/ops/logsoftmax.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/logsoftmax/cpu/logsoftmax_cpu.h  src/infiniop/ops/logsoftmax/cpu/../logsoftmax.h  src/infiniop/ops/logsoftmax/cpu/../../../operator.h  src/infiniop/ops/logsoftmax/cpu/../info.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils.h include/infinicore.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/custom_types.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/rearrange.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/result.hpp  src/infiniop/ops/logsoftmax/cpu/../../../../utils/check.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/../utils.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils/infini_status_string.h  src/infiniop/ops/logsoftmax/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/logsoftmax/cpu/../../../../utils.h\
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
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    }
}