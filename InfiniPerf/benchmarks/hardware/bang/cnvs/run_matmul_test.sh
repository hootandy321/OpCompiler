#!/bin/

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

if [ $# -ne 4 ]; then
    echo "Usage: $0 <data_type> <M> <K> <N>"
    echo "Example: $0 [fp32|fp16|int8|int16|tf32|bf16] 16384 16384 16384"
    exit 1
fi

DATA_TYPE=$1
DIM_M=$2
DIM_K=$3
DIM_N=$4

case "$DATA_TYPE" in
    fp32)
        in_type="float"
        out_type="float"
        ;;
    fp16)
        in_type="half"
        out_type="half"
        ;;
    int8)
        in_type="int8"
        out_type="half"
        echo "Using int8 input, set the output to half."
        ;;
    int16)
        in_type="int16"
        out_type="half"
        echo "Using int16 input, set the output to half."
        ;;
    tf32)
        in_type="tfloat32"
        out_type="tfloat32"
        ;;
    bf16)
        in_type="bfloat16"
        out_type="bfloat16"
        ;;
    *)
        echo "Error: Unknown input type '$DATA_TYPE'"
        echo "Supported types: fp32, fp16, int8, int16, tf32, bf16"
        exit 1
        ;;
esac

command -v cnvs >/dev/null 2>&1 || { 
    echo >&2 "Error: 'cnvs' command not found. Please source the environment or check installation."; 
    exit 1; 
}

BASE_DIR=$INFINIPERF_ROOT/benchmarks/hardware/bang/cnvs
CONFIG_PATH=${BASE_DIR}/matmul_${DATA_TYPE}_${DIM_M}x${DIM_K}x${DIM_N}.yaml

cat > "$CONFIG_PATH" <<EOF
custom:
- custom:
    matmul_performance:
      matrix_dimension_m: $DIM_M
      matrix_dimension_k: $DIM_K
      matrix_dimension_n: $DIM_N
      transpose_a: false
      transpose_b: true
      input_data_type: $in_type
      output_data_type: $out_type
      input_data_random: true
      correct_check: false
      iterations: 1000
EOF

echo "Performing Matmul test: $in_type, M = $DIM_M, K = $DIM_K, N = $DIM_N"

TMP_LOG=$(mktemp)

cnvs -c "$CONFIG_PATH" -r matmul_performance 2>&1 | tee "$TMP_LOG"

AVG_GOPS=$(awk '/GOPS\)/ { sum += $4; count++ } END { if (count > 0) printf("%.3f", sum / count); else print "0" }' "$TMP_LOG")
AVG_TOPS=$(awk -v g="$AVG_GOPS" 'BEGIN { printf("%.3f", g / 1000) }')

echo "----------------------------------------"
echo "Average Matmul performance: $AVG_TOPS TOPS"
echo "----------------------------------------"

rm -f "$TMP_LOG"
rm $CONFIG_PATH
rm -rf cnvs_stats/
