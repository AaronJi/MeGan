# export PIP_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/pypi/simple"
# export PIP_EXTRA_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/nvidia-pypi-remote-hz/simple"

# pip install transformers==4.46.1
# pip install accelerate==0.34.0

if [ -n "$MLP_WORKER_NUM" ]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=1
fi

cd /vepfs/group04/user/jl/projects/MetaLLamaFactory

OUT_MODEL_DIR="/vepfs/group04/user/jl/results/metaSwiglu/"

PROJECT_NAME="sft-test"
BASE_MODEL_NAME="Llama-3.2-1B-Instruct"
TYPE="full"
DATA="GYAFC_qa"
STYLE_TYPE="None"
MICRO_BATCH_SIZE=1
GRAD_ACC=1
LEARN_RATE=1e-4
TRAIN_EPOCH=1
BSZ=$((NNODES*GPUS_PER_NODE*MICRO_BATCH_SIZE*GRAD_ACC))
OUTPUT_MODEL_NAME=$PROJECT_NAME"-"$TYPE"-"$STYLE_TYPE"-"$BASE_MODEL_NAME"-"$DATA"-LR"$LEARN_RATE"-bsz"$BSZ"-epoch"$TRAIN_EPOCH

TRAINING_TYPE=$TYPE \
MODEL_NAME=$BASE_MODEL_NAME \
TOKENIZER_NAME=$BASE_MODEL_NAME \
OUTPUT_PATH=$OUT_MODEL_DIR$OUTPUT_MODEL_NAME \
DATASET=$DATA \
TEMPLATE="llama3" \
LR=$LEARN_RATE \
PER_DEVICE_BATCH_SIZE=$MICRO_BATCH_SIZE \
GRADIENT_ACCUMULATION_STEPS=$GRAD_ACC \
EPOCH=$TRAIN_EPOCH \
STYLE_PROMPT=$STYLE_TYPE \
PREPROCESS_WORKERS=1 \
sh train_meta_swish/run_llama_sft_template.sh
