#! /bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

if [ -z "$MEGATRON_ROOT" ]; then
    echo "Please make sure MEGATRON_ROOT has been correctly set: export MEGATRON_ROOT=/path/to/Megatron-LM"
    exit 1
fi

BASE_PATH=$MEGATRON_ROOT

LOG_PATH=${BASE_PATH}/log
rm -rf ${LOG_PATH}

DATA_CACHE_PATH=${BASE_PATH}/data_cache/${LOG_NAME}
rm -rf ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoints/${LOG_NAME}
rm -rf ${SAVE_PATH}
