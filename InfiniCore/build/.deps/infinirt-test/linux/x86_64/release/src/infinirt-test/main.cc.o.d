{
    files = {
        "src/infinirt-test/main.cc"
    },
    depfiles_format = "gcc",
    depfiles = "main.o: src/infinirt-test/main.cc src/infinirt-test/test.h  src/infinirt-test/../utils.h include/infinicore.h  src/infinirt-test/../utils/custom_types.h  src/infinirt-test/../utils/rearrange.h  src/infinirt-test/../utils/result.hpp src/infinirt-test/../utils/check.h  src/infinirt-test/../utils/../utils.h  src/infinirt-test/../utils/infini_status_string.h include/infinirt.h  include/infinicore.h\
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
            "-DNDEBUG"
        }
    }
}