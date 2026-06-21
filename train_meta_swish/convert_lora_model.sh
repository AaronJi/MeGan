PROJECT_DIR="/path/to/MeGan/"
#MODEL_NAME="llama_3.1_8b_instruct"
#LORA_MODEL_NAME="sft-lora-None-llama_3.1_8b_instruct-LR1e-4-bsz8-epoch1"
RESULT_MODEL_NAME="mergeModel-"$LORA_MODEL_NAME

MODEL_NAME=$1
LORA_MODEL_NAME=$2
RESULT_MODEL_NAME=$3

cd $PROJECT_DIR

LORA_MODEL_YAML="examples/merge_lora/"$LORA_MODEL_NAME".yaml"
cp examples/merge_lora/llama3_lora_sft_for_meta.yaml $LORA_MODEL_YAML

echo $MODEL_NAME
echo $LORA_MODEL_NAME
echo $RESULT_MODEL_NAME
echo $LORA_MODEL_YAML

sed -i 's/placeholder_basemodel/'$MODEL_NAME'/g' $LORA_MODEL_YAML
sed -i 's/placeholder_adaptor/'$LORA_MODEL_NAME'/g' $LORA_MODEL_YAML
sed -i 's/placeholder_resultmodel/'$RESULT_MODEL_NAME'/g' $LORA_MODEL_YAML

llamafactory-cli export $LORA_MODEL_YAML
