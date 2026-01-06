
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
    --do_train \
    --overwrite_output_dir \
    --overwrite_cache \
    --cutoff_len 4096 \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss \
    --bf16 \
    --ddp_timeout 360000000 \
    --max_samples 100000000 \
    --report_to tensorboard \
    --flash_attn fa2 \
"


: ${FINETUNE_TYPE:="meta_swiglu_simple"}
: ${TRAIN_STAGE:="sft"}
: ${MODEL_NAME:="llama_3.1_8b_instruct"}
: ${TOKENIZER_NAME:="llama_3.1_8b_instruct"}
: ${OUTPUT_PATH:="/vepfs/group04/user/jl/results/metaSwiglu/test_model"}
: ${DATASET:="Shakespeare_qa"}
: ${TEMPLATE:="llama3"}
: ${LR:=1e-4}
: ${LR_TYPE:="cosine"}
: ${WEIGHT_DECAY:=1e-2}
: ${PER_DEVICE_BATCH_SIZE:=1}
: ${GRADIENT_ACCUMULATION_STEPS:=1}
: ${EPOCH:=1}
: ${Meta_L2_LAMBDA:=0}
: ${SWIGLU_TYPE:="meta_swiglu"}
: ${STYLE_PROMPT:="None"}
: ${STYLE_EXPRESSION:="None"}
: ${META_SWISHGLU_ATTEN_KEY:="x"}
: ${META_SWISHGLU_ATTEN_HEAD_NUM:=4}
: ${META_SWISHGLU_MLP_BIAS:=1}
: ${META_SWISHGLU_BETA_N_LAYER:=1}
: ${PREPROCESS_WORKERS:=16}

MODEL_DIR="/vepfs/group04/beijing-public/models/"
MODEL_PATH=$MODEL_DIR$MODEL_NAME
TOKENIZER_PATH=$MODEL_DIR$MODEL_NAME

echo $TRAIN_STAGE
echo $DATASET
echo $TEMPLATE
echo $PREPROCESS_WORKERS
echo $LR, $PER_DEVICE_BATCH_SIZE, $GRADIENT_ACCUMULATION_STEPS, $EPOCH, $Meta_L2_LAMBDA
echo $MODEL_PATH
echo $OUTPUT_PATH
#exit 1

TRAIN_ARGS="
    --finetuning_type $FINETUNE_TYPE \
    --stage $TRAIN_STAGE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type $LR_TYPE \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --dataset  $DATASET \
    --template $TEMPLATE \
    --num_train_epochs $EPOCH \
    --style_prompt_type $STYLE_PROMPT \
    --style_expression_type $STYLE_EXPRESSION \
    --meta_swishglu_attn_key $META_SWISHGLU_ATTEN_KEY \
    --meta_swishglu_attn_head_num $META_SWISHGLU_ATTEN_HEAD_NUM \
    --meta_swishglu_mlp_bias $META_SWISHGLU_MLP_BIAS \
    --meta_swishglu_beta_num_layer $META_SWISHGLU_BETA_N_LAYER \
    --meta_swishglu_l2_lambda $Meta_L2_LAMBDA \
    --meta_hidden_act $SWIGLU_TYPE \
    --max_grad_norm  1.0 \
    --warmup_ratio 0.1 \
    --preprocessing_num_workers $PREPROCESS_WORKERS\
"
#     --meta_l2_lambda $Meta_L2_LAMBDA \
#     --meta_swishglu_l2_lambda $Meta_L2_LAMBDA \

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