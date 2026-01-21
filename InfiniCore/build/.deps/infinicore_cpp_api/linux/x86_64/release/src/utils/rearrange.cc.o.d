{
    files = {
        "src/utils/rearrange.cc"
    },
    depfiles_format = "gcc",
    depfiles = "rearrange.o: src/utils/rearrange.cc src/utils/rearrange.h  src/utils/result.hpp src/utils/check.h src/utils/../utils.h  include/infinicore.h src/utils/../utils/custom_types.h  src/utils/../utils/rearrange.h src/utils/infini_status_string.h\
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