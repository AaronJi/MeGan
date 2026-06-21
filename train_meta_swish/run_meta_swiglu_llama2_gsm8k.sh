#!/bin/bash

export TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1

cd /path/to/MetaLLamaFactory

OUTPUT_MODEL_PATH="/path/to/results/metaSwiglu/test"
MODEL_PATH="/path/to/models/Llama-2-7b-hf"
TOKENIZER_PATH="/path/to/models/Llama-2-7b-hf"
DATASET="gsm8k" 
OUTPUT_PATH=$OUTPUT_MODEL_PATH 
LR=1e-6 
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
EPOCH=3
TEMPLATE=llama2_v2
#FIREFLY="True"
USE_FAST_TOKENIZER="True"
Meta_L2_LAMBDA=1e-4
SWIGLU_TYPE="meta_swiglu"

source training_scripts/metaSwiGLU_llama2-7b_sft_template.sh
#bash training_scripts/metaSwiGLU_llama2-7b_sft_template.sh

rm -rf $OUTPUT_MODEL_PATH/checkpoint-*