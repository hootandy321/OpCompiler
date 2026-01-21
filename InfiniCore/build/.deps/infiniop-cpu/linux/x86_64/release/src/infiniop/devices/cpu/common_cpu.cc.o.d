{
    files = {
        "src/infiniop/devices/cpu/common_cpu.cc"
    },
    depfiles_format = "gcc",
    depfiles = "common_cpu.o: src/infiniop/devices/cpu/common_cpu.cc  src/infiniop/devices/cpu/common_cpu.h  src/infiniop/devices/cpu/../../../utils.h include/infinicore.h  src/infiniop/devices/cpu/../../../utils/custom_types.h  src/infiniop/devices/cpu/../../../utils/rearrange.h  src/infiniop/devices/cpu/../../../utils/result.hpp  src/infiniop/devices/cpu/../../../utils/check.h  src/infiniop/devices/cpu/../../../utils/../utils.h  src/infiniop/devices/cpu/../../../utils/infini_status_string.h  src/infiniop/devices/cpu/cpu_handle.h  src/infiniop/devices/cpu/../../handle.h include/infiniop/handle.h  include/infiniop/../infinicore.h\
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