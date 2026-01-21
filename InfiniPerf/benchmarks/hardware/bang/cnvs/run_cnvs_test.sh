#!/bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_type>"
    echo "Supported types: memory, pcie, mlulink"
    exit 1
fi

input_type="$1"

case "$input_type" in
    memory)
        type="memory_bandwidth"
        ;;
    pcie)
        type="pcie"
        ;;
    mlulink)
        type="mlulink"
        ;;
    *)
        echo "Error: Unknown test type '$input_type'"
        echo "Supported types: memory, pcie, mlulink"
        exit 1
        ;;
esac

BASE_DIR=$INFINIPERF_ROOT/benchmarks/hardware/bang/cnvs
CONFIG_DIR=${BASE_DIR}/cnvs_default_config

CONFIG_FILE=$(ls "${CONFIG_DIR}"/*.yml 2>/dev/null | head -n 1)

if [ -z "$CONFIG_FILE" ]; then
    bash $BASE_DIR/check_cnvs.sh
fi

echo "Using config file: $CONFIG_FILE to perform cnvs test."

cnvs -c $CONFIG_FILE -r $type

rm -rf cnvs_stats/
