#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "$MLP_WORKER_NUM" ]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=8
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
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 128 \
    --logging_steps 10 \
    --save_strategy epoch \
    --plot_loss \
    --bf16 \
    --weight_decay 1e-3 \
    --max_grad_norm 1.0 \
    --ddp_timeout 360000000 \
    --max_samples 100000000 \
    --report_to tensorboard \
    --flash_attn fa2 \
"

: ${MODEL_PATH:="/vepfs/DI/beijing-public/models/Qwen2-72B-Instruct"}
: ${TOKENIZER_PATH:="/vepfs/DI/beijing-public/models/Qwen2-72B-Instruct"}
: ${DATASET:="dataset"}
: ${OUTPUT_PATH:="output_path"}
: ${TEMPLATE:="xingrui_v3"}
: ${LR:=1e-6}
: ${LR_TYPE:="cosine"}
: ${PER_DEVICE_BATCH_SIZE:=4}
: ${GRADIENT_ACCUMULATION_STEPS:=2}
: ${EPOCH:=1}

TRAIN_ARGS="
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --output_dir $OUTPUT_PATH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type $LR_TYPE \
    --learning_rate $LR \
    --template $TEMPLATE \
    --dataset  $DATASET \
    --num_train_epochs $EPOCH \
"

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