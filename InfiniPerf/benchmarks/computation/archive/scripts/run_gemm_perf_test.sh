#!/bin/bash

if [ -z "$TEST_BINARY" ]; then
    echo "Please make sure InfiniCore has been correctly built by running: source InfiniCore/scripts/build_infinicore.sh"
    return 0
fi

case "$INFINIPERF_PLATFORM" in
    "NVIDIA_GPU")
        INFINIOP_TEST_OPTS="--nvidia"
        ;;
    "CAMBRICON_MLU")
        INFINIOP_TEST_OPTS="--cambricon"
        ;;
    "ASCEND_NPU")
        INFINIOP_TEST_OPTS="--ascend"
        ;;
    "METAX_GPU")
        INFINIOP_TEST_OPTS="--metax"
        ;;
    "MOORE_GPU")
        INFINIOP_TEST_OPTS="--moore"
        ;;
    "SUGON_DCU")
        INFINIOP_TEST_OPTS="--sugon"
        ;;
    "ILLUVATAR_GPU")
        INFINIOP_TEST_OPTS="--iluvatar"
        ;;
    *)
        echo "Unknown or unset INFINIPERF_PLATFORM: '$INFINIPERF_PLATFORM'"
        echo "Please make sure INFINIPERF_PLATFORM has been correctly set. "
        echo "  - Available options: ["NVIDIA_GPU", "CAMBRICON_MLU", "ASCEND_NPU", "METAX_GPU", "MOORE_GPU", "SUGON_DCU", "ILLUVATAR_GPU"]."
        return 0
        ;;
esac

function show_help() {
    echo "Usage: bash run.sh [options]"
    echo "Options:"
    echo "  --warmup <N>   Set the number of warmup runs before the test (default: 20)"
    echo "  --run <N>      Set the number of test runs (default: 1000)"
    echo "  --help         Show help message"
    exit 0
}

warmup=20
run=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --warmup requires a numeric value"
                show_help
            fi
            warmup=$2
            shift 2
            ;;
        --run)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --run requires a numeric value"
                show_help
            fi
            run=$2
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Error: Unknown parameter $1"
            show_help
            ;;
    esac
done

cd $TEST_CASE_PATH
python -m test_generate.testcases.gemm

output=$($TEST_BINARY ./gemm.gguf $INFINIOP_TEST_OPTS --warmup $warmup --run $run)

# 用于存储最终结果
final_output=""

# 解析 output
while IFS= read -r line; do
    # 保留原始输出
    final_output+="$line"$'\n'

    # 检测 GEMM 矩阵尺寸信息
    if [[ "$line" =~ "a: Shape:" ]]; then
        # 提取 M, K
        M=$(echo "$line" | awk -F'[][]' '{print $2}' | awk -F', ' '{print $1}')
        K=$(echo "$line" | awk -F'[][]' '{print $2}' | awk -F', ' '{print $2}')
    elif [[ "$line" =~ "b: Shape:" ]]; then
        # 提取 N
        N=$(echo "$line" | awk -F'[][]' '{print $2}' | awk -F', ' '{print $2}')
    elif [[ "$line" =~ "Time:" ]]; then
        # 提取时间 (单位: us)
        time_us=$(echo "$line" | grep -oP '(?<=Time: )[\d\.]+')

        # 计算 TFLOPS
        tflops=$(echo "scale=6; (2 * $M * $N * $K) / ($time_us * 1000000)" | bc -l)

        # 拼接 TFLOPS 信息
        final_output+="TFLOPS: $tflops"$'\n'
    fi
done <<< "$output"

cd -

# 最后一次性输出所有内容
echo "$final_output"
