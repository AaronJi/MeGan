#export PIP_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/pypi/simple"
#export PIP_EXTRA_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/nvidia-pypi-remote-hz/simple"

#pip install transformers==4.46.1
#pip install accelerate==0.34.0
#cd /path/to/MetaLLamaFactory
cd /vepfs/group04/user/jl/projects/MetaLLamaFactory

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
    --deepspeed ./examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --finetuning_type meta_swiglu_simple \
    --overwrite_output_dir \
    --overwrite_cache \
    --cutoff_len 4096 \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss \
    --bf16 \
    --weight_decay 1e-2 \
    --ddp_timeout 360000000 \
    --max_samples 100000000 \
    --report_to tensorboard \
    --flash_attn fa2 \
    --preprocessing_num_workers 1\
"

: ${MODEL_PATH:="/path/to/models/Llama-3.2-1B-Instruct"}
: ${TOKENIZER_PATH:="/path/to/models/Llama-3.2-1B-Instruct"}
: ${DATASET:="Shakespeare_qa"}
: ${OUTPUT_PATH:="/vepfs/group04/user/jl/results/metaSwiglu/metaSwiGLU_llama3_1b_sft_Shakespeare_qa_lr1e-4_0825"}
: ${TEMPLATE:="llama3"}
: ${LR:=1e-4}
: ${LR_TYPE:="cosine"}
: ${PER_DEVICE_BATCH_SIZE:=1}
: ${GRADIENT_ACCUMULATION_STEPS:=1}
: ${EPOCH:=1}
: ${Meta_L2_LAMBDA:=1e-4}
: ${SWIGLU_TYPE:="meta_swiglu"}

TRAIN_ARGS="
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type $LR_TYPE \
    --learning_rate $LR \
    --dataset  $DATASET \
    --template $TEMPLATE \
    --num_train_epochs $EPOCH \
    --meta_swiglu_l2_lambda $Meta_L2_LAMBDA \
    --meta_hidden_act $SWIGLU_TYPE \
    --max_grad_norm  1.0 \
    --warmup_ratio 0.1 \
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