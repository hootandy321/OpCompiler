#! /bin/bash

if [ -z "$INFINIPERF_ROOT" ]; then
    echo "Please make sure INFINIPERF_ROOT has been correctly set: export INFINIPERF_ROOT=/path/to/InfiniPerf"
    exit 1
fi

if [ -z "$MEGATRON_ROOT" ]; then
    echo "Please make sure MEGATRON_ROOT has been correctly set: export MEGATRON_ROOT=/path/to/Megatron-LM"
    exit 1
fi

if [ -z "$LLAMA2_7B_MODEL_PATH" ]; then
    echo "Please make sure LLAMA2_7B_MODEL_PATH has been correctly set: export LLAMA2_7B_MODEL_PATH=/path/to/Llama-2-7b-hf"
    exit 1
fi

if [ -e "$MEGATRON_ROOT/oscar_text_document.bin" ] && [ -e "$MEGATRON_ROOT/oscar_text_document.idx" ]; then
    echo "Oscar data has already been processed."
else
    echo "Please make sure oscar data has been correctly processed by running: bash process_data.sh"
    exit 1
fi

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Distributed training variables
NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost"

# Parallelism variables
TP=4
PP=2
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# Paths
BASE_PATH=$MEGATRON_ROOT
SRC_PATH=$BASE_PATH/pretrain_llama.py

LOG_NAME=llama2-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=${BASE_PATH}/oscar_text_document
DATA_CACHE_PATH=${BASE_PATH}/data_cache/${LOG_NAME}
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoints/${LOG_NAME}
mkdir -p ${BASE_PATH}/checkpoints/${LOG_NAME}

TOKENIZER_PATH=$LLAMA2_7B_MODEL_PATH/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       "    

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by 1 \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 32 \
       --train-iters 10000 \
       --log-interval 1 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --optimizer adam \
       --exit-interval 10 \
       --recompute-activations \
       --recompute-granularity selective \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       --save ${SAVE_PATH} \
       --save-interval 10 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 1000 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}