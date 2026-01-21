#!/bin/bash

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
        INFINIOP_TEST_OPTS="--Hygon"
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
    return 0
}

warmup=20
run=100

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

cd $INFINICORE_ROOT

output=$(python test/infinicore/ops/matmul.py --bench --nvidia --num_prerun $warmup --num_iterations $run)

final_output=$(echo "$output" | awk '
BEGIN {
    printf "%-40s %-10s %-18s %-15s\n", "Test shape (B,M,K,N)", "DType", "InfiniCore", "TFLOPs";
    print  "------------------------------------------------------------------------------";
}

$0 ~ /^TestCase\(Matmul - INPLACE\(out\)/ {
    B = M = K = N = 0;
    dtype = "N/A";

    if (match($0, /tensor\(([0-9]+), *([0-9]+), *([0-9]+)\)/, m)) {
        B = m[1] + 0;
        M = m[2] + 0;
        K = m[3] + 0;

        rest = substr($0, RSTART + RLENGTH);
        if (match(rest, /tensor\(([0-9]+), *([0-9]+), *([0-9]+)\)/, m2)) {
            N = m2[3] + 0;
        }
    }

    if (match($0, /, *([A-Za-z0-9_]+)[;)]/, dt)) { dtype = dt[1]; }

    case_desc = sprintf("B=%d, M=%d, K=%d, N=%d", B, M, K, N);
}

$0 ~ /InfiniCore time/ {
    if (match($0, /Device:[ ]*([0-9.]+)[ ]*ms/, t)) {
        dev_ms = t[1] + 0.0;

        if (B > 0 && M > 0 && K > 0 && N > 0) {
            flops  = B * 2.0 * M * N * K;
            tflops = flops / (dev_ms * 1e9);
            printf "%-40s %-10s %8.3f ms %10.2f TFLOPs\n", case_desc, dtype, dev_ms, tflops;
        } else {
            printf "%-40s %-10s %8.3f ms %10s\n", case_desc, dtype, dev_ms, "N/A";
        }
    }
}
' - )

cd -

echo "$final_output"
