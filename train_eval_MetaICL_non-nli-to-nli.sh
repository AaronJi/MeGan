if [ -n "$MLP_WORKER_NUM" ]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=1
fi

cd /path/to/MeGan/

OUT_MODEL_DIR="/path/to/MeGan/checkpoint/"

PROJECT_NAME="meta_swiglu"
BASE_MODEL_NAME="llama_3.1_8b_instruct"
STAGE="sft"
DATA="non_nli_to_nli"
STYLE_PROMPT_TYPE="inSystem"
STYLE_EXPRESSION_TYPE="withInstruction"
META_SHARED_HYPER=1
META_ATTN_KEY="x"
META_ATTN_HEAD_NUM=1
META_MLP_BIAS=0
META_NUM_LAYER=1
META_HIDDEN_DIM=512
META_START_LAYER=-1
LAMBDA=0
MICRO_BATCH_SIZE=1
GRAD_ACC=1
LEARN_RATE=1e-4
DECAY=1e-2
TRAIN_EPOCH=1
DATA_NAME=$DATA
BSZ=$((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE*GRAD_ACC))
OUTPUT_MODEL_NAME=$PROJECT_NAME"-s"$META_SHARED_HYPER"-attn_k_"$META_ATTN_KEY"_h"$META_ATTN_HEAD_NUM"_b"$META_MLP_BIAS"_l"$META_NUM_LAYER"-r"$META_HIDDEN_DIM"-sl"$META_START_LAYER"-"$STAGE"-"$BASE_MODEL_NAME"-"$DATA"0507-Lambda"$LAMBDA"-LR"$LEARN_RATE"-decay"$DECAY"-bsz"$BSZ"-epoch"$TRAIN_EPOCH

MODEL_NAME=$BASE_MODEL_NAME \
TOKENIZER_NAME=$BASE_MODEL_NAME \
OUTPUT_PATH=$OUT_MODEL_DIR$OUTPUT_MODEL_NAME \
TEMPLATE="llama3" \
STYLE_PROMPT=$STYLE_PROMPT_TYPE \
STYLE_EXPRESSION=$STYLE_EXPRESSION_TYPE \
FINETUNE_TYPE=$PROJECT_NAME \
TRAIN_STAGE=$STAGE \
DATASET=$DATA \
LR=$LEARN_RATE \
WEIGHT_DECAY=$DECAY \
PER_DEVICE_BATCH_SIZE=$MICRO_BATCH_SIZE \
GRADIENT_ACCUMULATION_STEPS=$GRAD_ACC \
EPOCH=$TRAIN_EPOCH \
META_SWISHGLU_SHARED_HYPER=$META_SHARED_HYPER \
META_SWISHGLU_ATTEN_KEY=$META_ATTN_KEY \
META_SWISHGLU_ATTEN_HEAD_NUM=$META_ATTN_HEAD_NUM \
META_SWISHGLU_MLP_BIAS=$META_MLP_BIAS \
META_SWISHGLU_BETA_N_LAYER=$META_NUM_LAYER \
META_SWISHGLU_BETA_HIDDEN_DIM=$META_HIDDEN_DIM \
META_SWISHGLU_START_LAYER_INDEX=$META_START_LAYER \
Meta_L2_LAMBDA=$LAMBDA \
PREPROCESS_WORKERS=16 \
sh train_meta_swish/run_meta_swiglu_sft_template.sh

EVAL_DIR="/path/to/MeGan/eval/"

if [ -n "$MLP_ROLE_INDEX" ]; then
  NODE_RANK="$MLP_ROLE_INDEX"
else
  NODE_RANK=0
fi


if [ $NODE_RANK -eq 0 ]; then
    STYLE_DOMAIN="nnli2nli-test"
    EVAL_NAME=$OUTPUT_MODEL_NAME"_prompt-"$STYLE_PROMPT_TYPE"_style-"$STYLE_EXPRESSION_TYPE"_domain-"$STYLE_DOMAIN
    python src/llamafactory/model/model_utils/evaluate_style.py --model_path $OUT_MODEL_DIR$OUTPUT_MODEL_NAME --output_dir $EVAL_DIR$EVAL_NAME --data_path /path/to/MeGan/data/metaicl/non_nli_to_nli/test_fix.jsonl --num_gpus 8 --style_prompt_type $STYLE_PROMPT_TYPE --style_expression_type $STYLE_EXPRESSION_TYPE --sample_format qa_task --style_domain $STYLE_DOMAIN --eval_form gen_choice --model_type 2
fi

if [ $NODE_RANK -eq 1 ]; then
    STYLE_DOMAIN="nnli2nli-unseen-test"
    EVAL_NAME=$OUTPUT_MODEL_NAME"_prompt-"$STYLE_PROMPT_TYPE"_style-"$STYLE_EXPRESSION_TYPE"_domain-"$STYLE_DOMAIN
    python src/llamafactory/model/model_utils/evaluate_style.py --model_path $OUT_MODEL_DIR$OUTPUT_MODEL_NAME --output_dir $EVAL_DIR$EVAL_NAME --data_path /path/to/MeGan/data/metaicl/non_nli_to_nli/unseen_test_fix.jsonl --num_gpus 8 --style_prompt_type $STYLE_PROMPT_TYPE --style_expression_type $STYLE_EXPRESSION_TYPE --sample_format qa_task --style_domain $STYLE_DOMAIN --eval_form gen_choice --model_type 2
fi