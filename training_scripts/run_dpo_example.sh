export PIP_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/pypi/simple"
export PIP_EXTRA_INDEX_URL="https://pkg.geely.com/artifactory/api/pypi/nvidia-pypi-remote-hz/simple"
export NCCL_DEBUG=VERSION
pip install vllm==0.5.0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/:$LD_LIBRARY_PATH
export NNODES=16

cd /path/to/LLaMA-Factory-0.8.2

OUTPUT_MODEL_PATH="/path/to/model/dpo_llama3_70b_lr5e-7"
MODEL_PATH="/path/to/model/sft_llama3_70b_lr1e-6_sharegpt_0901_base/checkpoint-603"
TOKENIZER_PATH="/path/to/model/sft_llama3_70b_lr1e-6_sharegpt_0901_base/checkpoint-603"
DATASET="CValues,huozi_rlhf_data,PKU-SafeRLHF-10K,HC3,human_test,dpo_zh_demo" 
OUTPUT_PATH=$OUTPUT_MODEL_PATH 
LR=5e-7 
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
TEMPLATE=llama3-chinese
USE_FAST_TOKENIZER="True"
EPOCH=3