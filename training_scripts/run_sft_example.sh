#!/bin/bash

export PIP_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/pypi/simple"
export PIP_EXTRA_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/nvidia-pypi-remote-hz/simple"

pip install transformers==4.44.0
pip install --upgrade datasets
pip install trl==0.9.6
pip install peft==0.12.0
pip install accelerate==0.32.0

export TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1

# 开始训练的时候改成LlamaFactory代码的公共目录或者个人git branch
cd /vepfs/DI/user/turghunrahman/workspace/LLaMA-Factory-0.8.2

OUTPUT_MODEL_PATH="/vepfs/DI/user/turghunrahman/model/sft/xxx"
MODEL_PATH="/vepfs/DI/beijing-public/models/Qwen2.5-7B-Instruct" 
TOKENIZER_PATH="/vepfs/DI/beijing-public/models/Qwen2.5-7B-Instruct" 
DATASET="merge_all_distinct" 
OUTPUT_PATH=$OUTPUT_MODEL_PATH 
LR=3e-7 
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
EPOCH=3
TEMPLATE=qwen
FIREFLY="True"
USE_FAST_TOKENIZER="True"
MULTI_INSTRUCTION="False"

source training_scripts/qwen7b_sft_template.sh

# rm -rf $OUTPUT_MODEL_PATH/checkpoint-*