{
    files = {
        "src/infiniop/reduce/cpu/reduce.cc"
    },
    depfiles_format = "gcc",
    depfiles = "reduce.o: src/infiniop/reduce/cpu/reduce.cc  src/infiniop/reduce/cpu/reduce.h  src/infiniop/reduce/cpu/../../../utils.h include/infinicore.h  src/infiniop/reduce/cpu/../../../utils/custom_types.h  src/infiniop/reduce/cpu/../../../utils/rearrange.h  src/infiniop/reduce/cpu/../../../utils/result.hpp  src/infiniop/reduce/cpu/../../../utils/check.h  src/infiniop/reduce/cpu/../../../utils/../utils.h  src/infiniop/reduce/cpu/../../../utils/infini_status_string.h\
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