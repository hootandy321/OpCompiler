#!/bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

if [ -z "$OSCAR_DATA_PATH" ]; then
    echo "Please make sure OSCAR_DATA_PATH has been correctly set: export OSCAR_DATA_PATH=/path/to/oscar-en-10k.jsonl"
    exit 1
fi

if [ -z "$LLAMA2_7B_MODEL_PATH" ]; then
    echo "Please make sure LLAMA2_7B_MODEL_PATH has been correctly set: export LLAMA2_7B_MODEL_PATH=/path/to/Llama-2-7b-hf"
    exit 1
fi

if [ -e "$MEGATRON_ROOT/oscar_text_document.bin" ] && [ -e "$MEGATRON_ROOT/oscar_text_document.idx" ]; then
    echo "Oscar data has already been processed."
else
    cd $MEGATRON_ROOT

    python ./tools/preprocess_data.py \
        --input $OSCAR_DATA_PATH \
        --output-prefix oscar \
        --workers 32 \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model $LLAMA2_7B_MODEL_PATH/tokenizer.model \
        --append-eod

    # This will generate 2 processed data file under $MEGATRON_ROOT, 
    # named oscar_text_document.bin & oscar_text_document.idx

    cd -
fi

