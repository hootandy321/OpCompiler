#!/bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    return 0
fi

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi
export INFINICORE_ROOT=$INFINIPERF_ROOT/benchmarks/computation/InfiniCore
export TEST_BINARY=$INFINICORE_ROOT/build/linux/${ARCH}/release/infiniop-test
export TEST_CASE_PATH=$INFINICORE_ROOT/test/infiniop-test

export INFINI_ROOT=~/.infini
export LD_LIBRARY_PATH=${INFINI_ROOT}/lib:${LD_LIBRARY_PATH}

export XMAKE_ROOT=y

. $INFINIPERF_ROOT/benchmarks/computation/scripts/install_prerequisites.sh

cp $INFINIPERF_ROOT/benchmarks/computation/scripts/testcases/* $INFINICORE_ROOT/test/infiniop-test/test_generate/testcases

cd $INFINICORE_ROOT
rm -rf build/
rm -rf .xmake/

case "$INFINIPERF_PLATFORM" in
    "NVIDIA_GPU")
        PLATFORM_OPTS="--nv-gpu=true --cuda=$CUDA_HOME"
        INFINIOP_TEST_OPTS="--nvidia"
        ;;
    "CAMBRICON_MLU")
        PLATFORM_OPTS="--cambricon-mlu=true --arch=$ARCH"
        ;;
    "ASCEND_NPU")
        PLATFORM_OPTS="--ascend-npu=true --arch=$ARCH"
        ;;
    "METAX_GPU")
        PLATFORM_OPTS="--metax-gpu=true --arch=$ARCH"
        ;;
    "MOORE_GPU")
        PLATFORM_OPTS="--moore-gpu=true --arch=$ARCH"
        ;;
    "SUGON_DCU")
        PLATFORM_OPTS="--sugon-dcu=true --arch=$ARCH"
        ;;
    "ILLUVATAR_GPU")
        PLATFORM_OPTS="--iluvatar-gpu=true --arch=$ARCH"
        ;;
    *)
        echo "Unknown or unset INFINIPERF_PLATFORM: '$INFINIPERF_PLATFORM'"
        echo "Please make sure INFINIPERF_PLATFORM has been correctly set. "
        echo "  - Available options: ["NVIDIA_GPU", "CAMBRICON_MLU", "ASCEND_NPU", "METAX_GPU", "MOORE_GPU", "SUGON_DCU", "ILLUVATAR_GPU"]."
        return 0
        ;;
esac

xmake f $PLATFORM_OPTS -cv
xmake build && xmake install && xmake build infiniop-test
cd -
