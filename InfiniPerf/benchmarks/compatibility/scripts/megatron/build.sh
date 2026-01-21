#!/bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

export OSCAR_DATA_PATH=/workspace/bolunz/Data/oscar-en-10k.jsonl
export LLAMA2_7B_MODEL_PATH=/workspace/bolunz/Models/Llama-2-7b-hf

export MEGATRON_ROOT=$INFINIPERF_ROOT/benchmarks/compatibility/Megatron-LM

cp $INFINIPERF_ROOT/benchmarks/compatibility/scripts/megatron/training.py $MEGATRON_ROOT/megatron/training
cp $MEGATRON_ROOT/pretrain_gpt.py $MEGATRON_ROOT/pretrain_llama.py

if ! python -c "import sentencepiece" &> /dev/null; then
    echo "sentencepiece 未安装，正在安装..."
    pip install sentencepiece
else
    echo "sentencepiece 已安装。"
fi