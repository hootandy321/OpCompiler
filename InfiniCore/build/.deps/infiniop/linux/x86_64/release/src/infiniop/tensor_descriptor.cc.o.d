{
    files = {
        "src/infiniop/tensor_descriptor.cc"
    },
    depfiles_format = "gcc",
    depfiles = "tensor_descriptor.o: src/infiniop/tensor_descriptor.cc  src/infiniop/../utils.h include/infinicore.h  src/infiniop/../utils/custom_types.h src/infiniop/../utils/rearrange.h  src/infiniop/../utils/result.hpp src/infiniop/../utils/check.h  src/infiniop/../utils/../utils.h  src/infiniop/../utils/infini_status_string.h src/infiniop/tensor.h  include/infiniop/tensor_descriptor.h include/infiniop/../infinicore.h\
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