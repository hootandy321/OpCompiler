{
    files = {
        "src/utils-test/main.cc"
    },
    depfiles_format = "gcc",
    depfiles = "main.o: src/utils-test/main.cc src/utils-test/utils_test.h  src/utils-test/../utils.h include/infinicore.h  src/utils-test/../utils/custom_types.h  src/utils-test/../utils/rearrange.h src/utils-test/../utils/result.hpp  src/utils-test/../utils/check.h src/utils-test/../utils/../utils.h  src/utils-test/../utils/infini_status_string.h\
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