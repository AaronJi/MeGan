#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "$MLP_WORKER_NUM" ]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=1
fi

if [ -n "$MLP_ROLE_INDEX" ]; then
  NODE_RANK="$MLP_ROLE_INDEX"
else
  NODE_RANK=0
fi

if [ -n "$MLP_WORKER_0_HOST" ]; then
  MASTER_ADDR="$MLP_WORKER_0_HOST"
  MASTER_PORT="$MLP_WORKER_0_PORT"
else
  MASTER_ADDR=localhost
  MASTER_PORT=12345
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

FIX_ARGS="
    --deepspeed ./examples/deepspeed/ds_z2_config.json \
    --do_train \
    --stage sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --logging_steps 10 \
    --save_strategy epoch \
    --plot_loss \
    --bf16 \
    --weight_decay 1e-3 \
    --ddp_timeout 360000000 \
    --max_samples 100000000 \
    --report_to tensorboard \
    --flash_attn fa2 \
"

: ${TRAINING_TYPE:="full"}
: ${MODEL_NAME:="llama_3.1_8b_instruct"}
: ${TOKENIZER_NAME:="llama_3.1_8b_instruct"}
: ${OUTPUT_PATH:="/vepfs/group04/user/jl/results/metaSwiglu/test_model"}
: ${DATASET:="Shakespeare_qa"}
: ${TEMPLATE:="llama3"}
: ${LR:=1e-4}
: ${LR_TYPE:="cosine"}
: ${PER_DEVICE_BATCH_SIZE:=4}
: ${GRADIENT_ACCUMULATION_STEPS:=2}
: ${EPOCH:=1}
: ${STYLE_PROMPT:="None"}
: ${PREPROCESS_WORKERS:=16}

MODEL_DIR="/vepfs/group04/beijing-public/models/"
MODEL_PATH=$MODEL_DIR$MODEL_NAME
TOKENIZER_PATH=$MODEL_DIR$MODEL_NAME

echo $TRAINING_TYPE
echo $DATASET
echo $TEMPLATE
echo $PREPROCESS_WORKERS
echo $LR, $PER_DEVICE_BATCH_SIZE, $GRADIENT_ACCUMULATION_STEPS, $EPOCH
echo $MODEL_PATH
echo $OUTPUT_PATH
#exit 1

TRAIN_ARGS="
    --finetuning_type $TRAINING_TYPE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type $LR_TYPE \
    --learning_rate $LR \
    --template $TEMPLATE \
    --dataset  $DATASET \
    --num_train_epochs $EPOCH \
    --style_prompt_type $STYLE_PROMPT \
    --preprocessing_num_workers $PREPROCESS_WORKERS\
"
# --tokenizer_name_or_path $TOKENIZER_PATH \

FLAG_AGRS=""
if [ -n "$USE_FAST_TOKENIZER" ]; then
  FLAG_AGRS="$FLAG_AGRS --use_fast_tokenizer"
fi
if [ -n "$FIREFLY" ]; then
  FLAG_AGRS="$FLAG_AGRS --firefly"
fi
if [ -n "$MULTI_INSTRUCTION" ]; then
  FLAG_AGRS="$FLAG_AGRS --multi_instructions"
fi

echo "### Final command:"
echo "torchrun $DISTRIBUTED_ARGS src/train.py $FIX_ARGS  $TRAIN_ARGS  $FLAG_AGRS"

mkdir -p $OUTPUT_PATH
torchrun $DISTRIBUTED_ARGS src/train.py $FIX_ARGS  $TRAIN_ARGS  $FLAG_AGRS