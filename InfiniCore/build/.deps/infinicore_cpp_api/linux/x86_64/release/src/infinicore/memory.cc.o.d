{
    files = {
        "src/infinicore/memory.cc"
    },
    depfiles_format = "gcc",
    depfiles = "memory.o: src/infinicore/memory.cc include/infinicore/memory.hpp  include/infinicore/device.hpp include/infinicore.h\
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