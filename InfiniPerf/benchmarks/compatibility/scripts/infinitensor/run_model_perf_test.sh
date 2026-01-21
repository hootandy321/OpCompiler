#!/bin/bash

# set -x
cuda_runner() {
    echo "Running GPU benchmark begin"
	local GPU_ID=$1

    echo "Testing On GPU $GPU_ID"
	for model_file in "$MODEL_DATA_PATH"/*.onnx; do
        model_name=$(basename "$model_file")
        echo "-$model_name-"
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$INFINIPERF_ROOT/benchmarks/compatibility/scripts/infinitensor/run_model.py" --model "$model_file" --device cuda --repeat 500
    done
    echo "Running GPU benchmark done"
}

mlu_runner() {
    echo "Running MLU benchmark..."
	local MLU_ID=$1

    echo "Testing On MLU $MLU_ID"
	for model_file in "$MODEL_DATA_PATH"/*.onnx; do
        model_name=$(basename "$model_file")
        echo "-$model_name-"
        MLU_VISIBLE_DEVICES=$MLU_ID python "$INFINIPERF_ROOT/benchmarks/compatibility/scripts/infinitensor/run_model.py" --model "$model_file" --device mlu --repeat 500
    done
    echo "Running MLU benchmark done"
}

ascend_runner() {
    echo "Running Ascend benchmark..."
	local NPU_ID=$1

    echo "Testing On Ascend $NPU_ID"
	for model_file in "$MODEL_DATA_PATH"/*.onnx; do
        model_name=$(basename "$model_file")
        echo "-$model_name-"
        ASCEND_RT_VISIBLE_DEVICES=$NPU_ID python "$INFINIPERF_ROOT/benchmarks/compatibility/scripts/infinitensor/run_model.py" --model "$model_file" --device npu --repeat 500
    done
    echo "Running Ascend benchmark done"
}

kunlun_runner() {
    echo "Running Kunlun benchmark..."
	local XPU_ID=$1

    echo "Testing On KUNLUN $XPU_ID"
	for model_file in "$MODEL_DATA_PATH"/*.onnx; do
        model_name=$(basename "$model_file")
        echo "-$model_name-"
        XPU_VISIBLE_DEVICES=$XPU_ID python "$INFINIPERF_ROOT/benchmarks/compatibility/scripts/infinitensor/run_model.py" --model "$model_file" --device kunlun --repeat 500
    done
    echo "Running Ascend benchmark done"
}

# 准备数据集
prepare_datasets() {
	if [ -z "$MODEL_DATA_PATH" ]; then
	    echo "Please make sure MODEL_DATA_PATH has been correctly set: export MODEL_DATA_PATH=/path/to/InfiniPerfModels"
    	exit 1
	fi
    local dataset_target="$INFINIPERF_ROOT/benchmarks/compatibility/datasets"

    # 如果目标目录不存在则创建
    if [ ! -d "$dataset_target" ]; then
        mkdir -p "$dataset_target"
    fi

    # 检查是否已经链接
    if [ ! -L "$dataset_target/InfiniPerfModels" ]; then
        echo "Creating symlink to datasets..."
        ln -s "$MODEL_DATA_PATH" "$dataset_target/InfiniPerfModels"
    else
        echo "Dataset symlink already exists"
    fi

    # 验证数据集是否可用
    if [ ! -e "$dataset_target/InfiniPerfModels" ]; then
        echo "Error: Failed to prepare datasets!"
        exit 1
    fi
	MODEL_DATA_PATH="${dataset_target}/InfiniPerfModels"
}

# 主函数
main() {
    # 参数检查
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <device> <card_id>"
        echo "Available devices: gpu, mlu, npu, xpu"
		echo "Example: $0 gpu 0"
        exit 1
    fi

    local device=$1
	local card_id=$2

	if [ -z "$INFINIPERF_ROOT" ]; then
    	echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
	    exit 1
	fi

    # 准备数据集
    prepare_datasets
	echo "MODEL_DATA_PATH after function: $MODEL_DATA_PATH"
    # 根据设备类型执行对应函数
    case "$device" in
        gpu)
            cuda_runner "$card_id"
            ;;
        mlu)
            mlu_runner "$card_id"
            ;;
        npu)
            ascend_runner "$card_id"
            ;;
        xpu)
            kunlun_runner "$card_id"
            ;;
        *)
            echo "Error: Unknown device '$device'"
            echo "Available devices: gpu, mlu, npu, xpu"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
