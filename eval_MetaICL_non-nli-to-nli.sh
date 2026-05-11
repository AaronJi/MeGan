cd /path/to/MeGan/

STYLE_PROMPT_TYPE="inSystem"
STYLE_EXPRESSION_TYPE="withInstruction"

OUT_MODEL_DIR="/path/to/MeGan/checkpoint/"
OUTPUT_MODEL_NAME="the_model_checkpoint_name"

EVAL_DIR="/path/to/MeGan/eval/"
STYLE_DOMAIN="nnli2nli-test"
EVAL_NAME=$OUTPUT_MODEL_NAME"_prompt-"$STYLE_PROMPT_TYPE"_style-"$STYLE_EXPRESSION_TYPE"_domain-"$STYLE_DOMAIN
python src/llamafactory/model/model_utils/evaluate_style.py --model_path $OUT_MODEL_DIR$OUTPUT_MODEL_NAME --output_dir $EVAL_DIR$EVAL_NAME --data_path /path/to/MeGan/data/metaicl/non_nli_to_nli/test_fix.jsonl --num_gpus 1 --style_prompt_type $STYLE_PROMPT_TYPE --style_expression_type $STYLE_EXPRESSION_TYPE --sample_format qa_task --style_domain $STYLE_DOMAIN --eval_form gen_choice --model_type 2