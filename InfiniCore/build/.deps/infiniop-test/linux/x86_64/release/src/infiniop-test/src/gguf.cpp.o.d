{
    files = {
        "src/infiniop-test/src/gguf.cpp"
    },
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
            "-I/home/qy/.infini/include",
            "-I/home/qy/src/Infini/InfiniCore/src/infiniop-test/include",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fopenmp",
            "-DNDEBUG"
        }
    },
    depfiles_format = "gcc",
    depfiles = "gguf.o: src/infiniop-test/src/gguf.cpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/gguf.hpp  /home/qy/src/Infini/InfiniCore/src/infiniop-test/include/file_mapping.hpp\
"
}