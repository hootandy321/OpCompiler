{
    files = {
        "src/infiniop/ops/rms_norm/operator.cc"
    },
    depfiles_format = "gcc",
    depfiles = "operator.o: src/infiniop/ops/rms_norm/operator.cc  src/infiniop/ops/rms_norm/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/rms_norm/../../handle.h include/infiniop/handle.h  include/infiniop/ops/rms_norm.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/rms_norm/cpu/rms_norm_cpu.h  src/infiniop/ops/rms_norm/cpu/../rms_norm.h  src/infiniop/ops/rms_norm/cpu/../../../operator.h  src/infiniop/ops/rms_norm/cpu/../info.h  src/infiniop/ops/rms_norm/cpu/../../../../utils.h include/infinicore.h  src/infiniop/ops/rms_norm/cpu/../../../../utils/custom_types.h  src/infiniop/ops/rms_norm/cpu/../../../../utils/rearrange.h  src/infiniop/ops/rms_norm/cpu/../../../../utils/result.hpp  src/infiniop/ops/rms_norm/cpu/../../../../utils/check.h  src/infiniop/ops/rms_norm/cpu/../../../../utils/../utils.h  src/infiniop/ops/rms_norm/cpu/../../../../utils/infini_status_string.h  src/infiniop/ops/rms_norm/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/rms_norm/cpu/../../../../utils.h\
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