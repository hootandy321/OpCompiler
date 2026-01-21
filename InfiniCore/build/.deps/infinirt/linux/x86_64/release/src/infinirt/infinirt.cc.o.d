{
    files = {
        "src/infinirt/infinirt.cc"
    },
    depfiles_format = "gcc",
    depfiles = "infinirt.o: src/infinirt/infinirt.cc include/infinirt.h  include/infinicore.h src/infinirt/../utils.h include/infinicore.h  src/infinirt/../utils/custom_types.h src/infinirt/../utils/rearrange.h  src/infinirt/../utils/result.hpp src/infinirt/../utils/check.h  src/infinirt/../utils/../utils.h  src/infinirt/../utils/infini_status_string.h  src/infinirt/ascend/infinirt_ascend.h  src/infinirt/ascend/../infinirt_impl.h src/infinirt/bang/infinirt_bang.h  src/infinirt/bang/../infinirt_impl.h src/infinirt/cpu/infinirt_cpu.h  src/infinirt/cpu/../infinirt_impl.h src/infinirt/cuda/infinirt_cuda.cuh  src/infinirt/cuda/../infinirt_impl.h  src/infinirt/kunlun/infinirt_kunlun.h  src/infinirt/kunlun/../infinirt_impl.h  src/infinirt/metax/infinirt_metax.h  src/infinirt/metax/../../infiniop/devices/metax/metax_ht2mc.h  src/infinirt/metax/../infinirt_impl.h  src/infinirt/moore/infinirt_moore.h  src/infinirt/moore/../infinirt_impl.h\
",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
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
            "-DNDEBUG"
        }
    }
}