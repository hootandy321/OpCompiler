{
    files = {
        "src/infiniccl/infiniccl.cc"
    },
    depfiles_format = "gcc",
    depfiles = "infiniccl.o: src/infiniccl/infiniccl.cc include/infiniccl.h  include/infinirt.h include/infinicore.h  src/infiniccl/./ascend/infiniccl_ascend.h  src/infiniccl/./ascend/../infiniccl_impl.h  src/infiniccl/./cambricon/infiniccl_cambricon.h  src/infiniccl/./cambricon/../infiniccl_impl.h  src/infiniccl/./cuda/infiniccl_cuda.h  src/infiniccl/./cuda/../infiniccl_impl.h  src/infiniccl/./kunlun/infiniccl_kunlun.h  src/infiniccl/./kunlun/../infiniccl_impl.h  src/infiniccl/./metax/infiniccl_metax.h  src/infiniccl/./metax/../../infiniop/devices/metax/metax_ht2mc.h  src/infiniccl/./metax/../infiniccl_impl.h  src/infiniccl/./moore/infiniccl_moore.h  src/infiniccl/./moore/../infiniccl_impl.h\
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