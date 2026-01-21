{
    files = {
        "src/infinicore/context/allocators/host_allocator.cc"
    },
    depfiles_format = "gcc",
    depfiles = "host_allocator.o: src/infinicore/context/allocators/host_allocator.cc  src/infinicore/context/allocators/host_allocator.hpp  src/infinicore/context/allocators/memory_allocator.hpp  include/infinicore/memory.hpp include/infinicore/device.hpp  include/infinicore.h include/infinirt.h include/infinicore.h\
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