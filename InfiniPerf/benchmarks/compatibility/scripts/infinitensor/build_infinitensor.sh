#!/bin/bash

# 检查CMake版本是否>=3.18
check_cmake_version() {
    current_version=$(cmake --version | grep -oP '(?<=version )[\d.]+')

    if [ -z "$current_version" ]; then
        echo "错误：未检测到CMake安装"
        exit 1
    fi

    required_version="3.18"

    if ! printf '%s\n' "$required_version" "$current_version" | sort -V -C 2>/dev/null; then
        echo "错误：CMake版本过低 (当前版本: $current_version, 需要版本: >= $required_version)"
        exit 1
    fi
}

# 根据硬件平台执行编译安装
install_for_platform() {
    local platform=$1
    local build_dir=$INFINITENSOR_ROOT

    case "$platform" in
        gpu)
            cmake_option="CUDA=ON"
            ;;
        mlu)
            cmake_option="BANG=ON"
            ;;
        npu)
            cmake_option="ASCEND=ON"
            ;;
        xpu)
            cmake_option="KUNLUN=ON"
            ;;
        *)
            echo "错误：未知硬件平台 '$platform'"
            echo "可用选项: gpu, mlu, npu, xpu"
            exit 1
            ;;
    esac

    if [ ! -d "$build_dir" ]; then
        echo "错误：构建目录 '$build_dir' 不存在"
        exit 1
    fi

    echo "正在为 $platform 平台安装Python绑定..."
    echo "使用选项: $cmake_option"

    cd "$build_dir" || exit 1
	git submodule update --init
    make install-python "$cmake_option"
    cd -
}

# 主函数
main() {
    # 检查参数
    if [ $# -ne 1 ]; then
        echo "用法: $0 <platform>"
        echo "可用平台选项: gpu, mlu, npu, xpu"
        exit 1
    fi

	if [ -z "$INFINIPERF_ROOT" ]; then
    	echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
	    exit 1
	fi
	export INFINITENSOR_ROOT=$INFINIPERF_ROOT/benchmarks/compatibility/InfiniTensor

    # 检查CMake版本
    check_cmake_version

    # 执行安装
    install_for_platform "$1"
}

# 执行主函数
main "$@"
