#!/bin/bash
set -e

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

BASE_DIR=$INFINIPERF_ROOT/benchmarks/hardware/bang/cnvs

command -v cnvs >/dev/null 2>&1 || { 
    echo >&2 "Error: 'cnvs' command not found. Please source the environment or check installation."; 
    exit 1; 
}

echo "Generating default cnvs config..."
cd $BASE_DIR
cnvs -y
cd - 
