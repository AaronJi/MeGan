import os
import json
import torch
import re
from collections import Counter
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from meta_swiglu_modeling_llama import LlamaForCausalLM
from meta_swiglu_shared_modeling_llama import LlamaForCausalLM_sharedHyper
#from ..model.model_utils.meta_swiglu_modeling_llama import LlamaForCausalLM
from safetensors.torch import load_file
import gc
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import rouge
from rouge import Rouge
import nltk
import time
from torch.nn.parallel import DataParallel
import logging
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import argparse
import hashlib
from pathlib import Path
from rouge_score import rouge_scorer
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, OrderedDict
from evalplus.eval import  estimate_pass_at_k, untrusted_check

# Set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# nltk.download('punkt', quiet=True)

def load_safetensors_checkpoint(model, checkpoint_path):
    """
    Load model parameter from shareded safetensors

    Args:
        model: init weights
        checkpoint_path: model checkpoint path

    Returns:
        loaded_beta_weights: parameters of beta_generator
    """
    safetensors_files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')])
    assert len(safetensors_files) > 0

    loaded_beta_weights = {}

    for filename in safetensors_files:
        file_path = os.path.join(checkpoint_path, filename)
        logger.info(f"Loading {filename}...")

        state_dict = load_file(file_path)

        for name, param in state_dict.items():
            if "beta_generator" in name or "style_attention" in name or "layer_embeddings" in name:
                loaded_beta_weights[name] = param.clone()

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return loaded_beta_weights

def load_bin_checkpoint(model, checkpoint_path):

    state_dict: Dict[str, torch.Tensor] = OrderedDict()

    loaded_beta_weights = {}

    for filepath in tqdm(os.listdir(checkpoint_path), desc="Load weights"):
        if os.path.isfile(os.path.join(checkpoint_path, filepath)) and filepath.endswith(".bin") and filepath.startswith("pytorch_model"):
            shard_weight = torch.load(os.path.join(checkpoint_path, filepath), map_location="cpu")
            state_dict.update(shard_weight)

    for name, param in state_dict.items():
        if "beta_generator" in name or "style_attention" in name or "layer_embeddings" in name:
            loaded_beta_weights[name] = param.clone()

    model_dict = model.state_dict()

    #matched_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    matched_dict = state_dict

    model_dict.update(matched_dict)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    if missing_keys:
        print(f"Warning, missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning, unexpected keys: {unexpected_keys}")

    #print("****")
    #print(loaded_beta_weights.keys())

    #print('###')
    #print(model_dict['model.shared_hyper_style2beta.style_attention.attention.in_proj_weight'])
    #print('###')
    #print(model_dict['model.layers.0.mlp.hyper_style2beta.style_attention.attention.in_proj_weight'])
    #print('###')
    #print(model_dict['model.layers.10.mlp.hyper_style2beta.style_attention.attention.in_proj_weight'])
    #exit(5)

    return loaded_beta_weights

def rl_eval(references: list, predictions: list):
    """
    Compute Rouge-L score

    Args:
        ground_truth: ref response
        predictions: gen response

    Returns:
        rougeL_fmeasure: Rouge-L F1 score
    """
    # 使用rouge_score库替代原来的rouge库
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    return sum(scores) / len(scores)


def b2_eval(references: list, predictions: list):
    weights = [0.5, 0.5]
    smoothie = SmoothingFunction().method3

    refs = [tmp.split(' ') for tmp in references]
    hyps = [tmp.split(' ') for tmp in predictions]
    scores = [sentence_bleu([reference], hypothesis, weights=weights, smoothing_function=smoothie) for reference, hypothesis in zip(refs, hyps)]
    return sum(scores) / len(scores)

def distinct_n(seqs, n=2):
    grams = [g for s in seqs for g in zip(*[s[i:] for i in range(n)])]
    score = len(set(grams)) / max(len(grams), 1)
    return score

def zipf_coefficient(seqs):
    """
    The slope of log(rank) ~ log(freq)
    more natural if closer to -1
    """
    words = [w for s in seqs for w in s.split()]
    freq = Counter(words)
    if len(freq) < 10:
        return 0.0
    ranks, freqs = zip(*enumerate(sorted(freq.values(), reverse=True), 1))
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    coeff = np.polyfit(log_r, log_f, 1)[0]
    return coeff


def macro_F1():

    return

def get_style_expression(style, instruction=None, style_expression_type="withInstruction", for_prompt=False):
    assert style_expression_type in ["None", "withInstruction"]
    style_expression = style
    if style_expression_type == "None":
        if isinstance(style, List):
            style_expression = '\n'.join(style)
        return style_expression

    # SST
    if style in ['very positive', 'positive', 'neutral', 'negative', 'very negative']:
        # style_expression = f"Please answer the question with the sentiment of **{style}**."
        # style_expression = f"Please answer the question with following sentiment.\n\n **sentiment**: {style}"
        # style_expression = f"Response sentiment: {style}"
        style_expression = f"Reply with sentiment: {style}"
        return style_expression

    if instruction and len(instruction) > 0:
        # Persona-Chat
        if 'profile' in instruction:
            #instruction = 'You are engaged in a conversation with the user. The user have the following profiles. You should consider the profiles and make the corresponding appropriate response.'
            #instruction = 'You are chatting with a user, who has the following profiles. Consider the profiles and respond.'
            #instruction = 'Provide the appropriate response based on the user profiles.'
            style_expression = instruction + '\n\nProfiles: \n'
            for profile in style:
                style_expression += profile + '\n'
        # AdaptSum
        elif 'a concise summary' in instruction:
            #instruction = 'You are dealing with an abstractive summarization with different domains. You should conside the following domain tag, and adjust your summarization accordingly.'
            #instruction = 'You are dealing with an abstractive summarization with different domains. Adjust your summarization accordingly.'
            #instruction = 'Conduct the summarization based its specific domains.'
            style_expression = instruction + '\n\nDomain: ' + style
        else:
            # MetaICL, SNI
            if for_prompt:
                style_expression = instruction
            else:
                style_expression = style
    else:
        # TODO general style
        style_expression = f"Please answer the question with the style of {style}."

        # SNI
        if for_prompt:
            style_expression = None
        else:
            style_expression = style

    return style_expression

def merge_history(history):
    if history is None or len(history) == 0:
        return None
    if isinstance(history, List):
        history_str = ''
        for turn in history:
            q = turn[0]
            a = turn[1]
            user_str = 'user: ' + q
            assistant_str = 'assistant: ' + a
            history_str += user_str + '\n' + assistant_str + '\n'
    else:
        history_str = history
    return history_str

def inference_worker(rank, data_path, model_path, output_dir, args, max_samples=None, max_new_tokens=128, total_workers=8, style_prompt_type="None", style_expression_type="None"):
    """
    多卡推理的工作进程函数

    Args:
        rank: the process index
        data_path: data file path
        model_path: model file path
        output_dir: output directory
        max_new_tokens: max generation length
        total_workers: the total number of workers
    """
    # Set the available GPU of the current process
    start_gpu = rank
    gpu_list = [start_gpu]
    gpu_list_str = ','.join(map(str, gpu_list))
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str
    device = torch.device(f'cuda:{rank}')
    output_file = Path(output_dir) / f'inference_result_part_{rank}.jsonl'

    # load dataset
    questions, references, styles, instructions, histories, task_names = load_dataset(data_path, args=args, max_samples=max_samples)
    all_tasks = []
    for task_name in task_names:
        if task_name and task_name not in all_tasks:
            all_tasks.append(task_name)
    num_available_subtasks = len(all_tasks)

    # split data
    data_tuples = list(zip(questions, references, styles, instructions, histories, task_names, range(len(questions))))
    data_slice = np.array_split(data_tuples, total_workers)[rank]

    # load existed results
    existing_results = {}
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # 使用问题和风格生成唯一键
                    input_hash = hashlib.md5((data["question"] + data["style"]).encode()).hexdigest()
                    existing_results[input_hash] = data
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")

    # initialize model and tokenizer
    #is_meta = 'meta_swiglu-' in model_path
    if args.model_type > 0:
        config = transformers.AutoConfig.from_pretrained(model_path)
        if args.model_type > 1:
            logger.info(f"Worker {rank} loading model with shared hypernetwork...")
            model = LlamaForCausalLM_sharedHyper(config)
            # TODO add bin load
            loaded_beta_weights = load_bin_checkpoint(model, model_path)
        else:
            logger.info(f"Worker {rank} loading model with layer-wise hypernetwork...")
            model = LlamaForCausalLM(config)
            loaded_beta_weights = load_safetensors_checkpoint(model, model_path)
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        logger.info(f"Worker {rank} loading model in the normal format...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16) # , attn_implementation='flash_attention_2'

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # processing data by samples
    for i, (question, reference, style, instruction, history, task_name, original_idx) in enumerate(tqdm(data_slice, desc=f"Worker {rank}")):
        # check if processed
        #input_hash = hashlib.md5((question + style).encode()).hexdigest()
        #if input_hash in existing_results:
        #    continue

        if style is None:
            # skip if no style provided
            continue

        try:
            #instruction = f"Please answer the question with the style of {style}."

            system = None
            assert style_prompt_type in ["None", "inSystem", "inQuery"]

            system = "" if system is None or len(system) == 0 else system
            if style_prompt_type != "None":
                style_expression = get_style_expression(style, instruction, for_prompt=True)
                if style_prompt_type == "inSystem":
                    system += style_expression
                elif style_prompt_type == "inQuery":
                    if 'profile' in instruction:
                        question = style_expression + "\n\nQuestion:\n" + question
                    else:
                        question += "\n\n" + style_expression

            # append history
            if args.eval_form !="gen_code" and history and len(history) > 0:
                history_str = merge_history(history)
                # TODO need to optimize
                if isinstance(history, List):
                    # Persona-Chat
                    history_str = '\nConversation history:\n' + history_str
                    system += history_str
                else:
                    # MetaICL
                    history_str = '\nConsider context:\n' + history_str + "\n\n"
                    question = history_str + "\nQuestion:\n" + question

            # Process user message by apply_chat_template
            user_message = [{"role": "user", "content": question}]
            if system is None or len(system) > 0:
                user_message.insert(0, {"role": "system", "content": system})

            chat_input = tokenizer.apply_chat_template(user_message, return_tensors="pt", add_generation_prompt=True)

            # encode style text
            style_expression = get_style_expression(style, instruction, style_expression_type)
            style_input = tokenizer(style_expression, return_tensors="pt", add_special_tokens=False)["input_ids"]

            # Move to GPU
            chat_input = chat_input.to(device)
            style_input = style_input.to(device)

            # inference the single sample
            with torch.no_grad():
                if args.model_type > 0:
                    gen_ids = model.generate(
                        input_ids=chat_input,
                        style=style_input,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        use_cache=True
                    )
                else:
                    gen_ids = model.generate(
                        input_ids=chat_input,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        use_cache=True
                    )

            # decode result, skip input and leave generation
            response = tokenizer.decode(
                gen_ids[0][len(chat_input[0]):],
                skip_special_tokens=True
            )

            # create result record
            record = {
                "index": original_idx,
                "system": system,
                "question": question,
                "style": style_expression,
                "reference": reference,
                "response": response
            }

            if args.eval_form =="gen_code":
                record["problem"] = history

            #print("&&&&&&&")
            #print(record)
            #exit(5)
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"Error processing sample {original_idx}: {e}")
            # save error record
            error_record = {
                "index": original_idx,
                "question": question,
                "style": style,
                "reference": reference,
                "response": f"ERROR: {str(e)}"
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    logger.info(f"Worker {rank} finished processing {len(data_slice)} samples")


def load_dataset(data_path, args, max_samples=None):
    """
    Args:
        data_path:
        max_samples:

    Returns:
        questions:
        references:
        styles:
    """
    # data_format = 'json'
    data_format = 'jsonl'

    queries = []
    responses = []
    styles = []
    instructions = []
    histories = []
    task_names = []
    #print("data_path，", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        if data_format == 'jsonl':
            samples = []
            for i, line in enumerate(f):
                try:
                    if max_samples is not None and i >= max_samples:
                        break
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except Exception as err:
                    import logging
                    logging.exception(err)
        else:
            samples = json.load(f)  # list of samples
            if max_samples is not None:
                samples = samples[:max_samples]

        for sample in samples:
            query, response, style, instruction, history = extract_sample(sample, args)
            queries.append(query)
            responses.append(response)
            styles.append(style)
            instructions.append(instruction)
            histories.append(history)
            if 'task_name' in sample:
                task_names.append(sample['task_name'])
            else:
                task_names.append(None)
    return queries, responses, styles, instructions, histories, task_names

def extract_sample(sample, args):
    # sample_format = 'text_style_role'
    # sample_format = 'qa_style_role'
    #sample_format = 'qa_style'

    query = None
    response = None
    style = None
    instruction = None
    history = None
    if args.sample_format == 'text_style_role':
        query = sample["messages"][0]["content"]
        response = None
        style = sample["messages"][1]["content"]
    elif args.sample_format == 'qa_style_role':
        query = sample['messages'][0]['content']
        response = sample['messages'][1]['content']
        style = sample['messages'][2]['content']
    elif args.sample_format == 'qa_style':
        query = sample['messages'][0]['content']
        response = sample['messages'][-1]['content']
        style = sample['messages'][-1]['style']
    elif args.sample_format == 'qa_sentiment':
        query = sample['messages'][0]['content']
        response = sample['messages'][-1]['content']
        style = sample['messages'][-1]['sentiment']
    elif args.sample_format == 'qa_domain':
        query = sample['messages'][0]['content']
        response = sample['messages'][-1]['content']
        instruction = sample['messages'][0]['instruction']
        style = sample['domain']
    elif args.sample_format == 'qa_task':
        query = sample['messages'][0]['content']
        response = sample['messages'][-1]['content']
        instruction = sample['messages'][0].get('instruction', None)
        style = sample['task_description']
        history = sample.get('context', None)
    elif args.sample_format == 'qa_persona':
        query = sample['messages'][0]['content']
        response = sample['messages'][-1]['content']
        instruction = sample['messages'][0]['instruction']
        style = sample['user_profiles']
        history = sample['history']
    return query, response, style, instruction, history

def get_classification_results(references, predictions, mode="include"):
    if_correct = []
    for ref, pred in zip(references, predictions):
        c = 0
        #if ref in pred:

        # mode = include, case_include, strict, case_strict
        if mode == "include":
            if ref.lower() in pred.lower():
                c = 1
        if mode == "case_include":
            if ref in pred:
                c = 1
        if mode == "strict":
            if ref.lower() == pred.lower():
                c = 1
        if mode == "case_strict":
            if ref == pred:
                c = 1

        if_correct.append(c)
    return if_correct

def extract_match_answer_number(completion: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))

def get_exact_number(s):
    s = s.strip()
    answer_prefix = "#### "
    if answer_prefix in s:
        s = s.split(answer_prefix)[1]
    num = extract_match_answer_number(s)
    return num

def get_math_results(references, predictions):
    if_correct = []
    for ref, pred in zip(references, predictions):
        ans_pred = get_exact_number(pred)
        ans_ref = get_exact_number(ref)
        try:
            if ans_pred and int(ans_pred) == int(ans_ref):
                if_correct.append(1)
            else:
                if_correct.append(0)
        except:
            if_correct.append(0)

        #print("anwser ,", infer_res, "reference ,", ref, "if correct ", int(infer_res)==int(ref))
    return if_correct

def get_coding_results(references, predictions, problems):
    evaluate_res = []
    for ref, code,problem in zip(references, predictions,problems):
        #print("task_id", problem["task_id"])
        dataset = None
        if "humaneval" in problem["task_id"].lower():
            dataset = "humaneval"
        if "mbpp" in problem["task_id"].lower():
            dataset = "mbpp"
        assert  dataset is not None
        # implement code_run; use evalplus libary; refer to J.1.2, Page 20 in Text-to-LoRA
        # extract_code from generations
        try:
            code = code.split("```python")[1].split("```")[0]
        except Exception as err:
            logger.info(f"task_id {problem['task_id']} code format error!!")
            pass
        ret = {"task_id": problem["task_id"], "solution": code,}
        #  problem
        ret["base"] = untrusted_check(dataset,
                                code,
                                problem["base_input"],
                                problem["entry_point"],
                                expected=ref["base"],
                                atol=problem["atol"],
                                ref_time=ref["base_time"],
                                fast_check=False,
                                min_time_limit=0.1,
                                gt_time_limit_factor=2.0,
                                )
        evaluate_res.append(ret)
    return evaluate_res


def calculate_coding_eval_res(evaluate_res):

    total = np.array([1]*len(evaluate_res))
    base_correct = []
    # plus_correct = []

    for ele in evaluate_res:
        base_correct.append(ele["base"][0]=="pass")
    base_correct_array = np.array(base_correct)
    base_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, base_correct_array, k).mean() for k in [1, 10, 100] if total.min() >= k}

    return base_pass_at_k

def merge_results(output_dir, total_workers):
    """

    Args:
        output_dir:
        total_workers:

    Returns:
        all_results:
    """
    output_dir = Path(output_dir)
    all_results = []

    for rank in range(total_workers):
        part_file = output_dir / f'inference_result_part_{rank}.jsonl'
        if part_file.exists():
            try:
                with open(part_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line.strip()))
            except Exception as e:
                logger.error(f"Error loading part {rank}: {e}")

    # sort by index
    # all_results.sort(key=lambda x: x["index"])

    # save merged results
    #merged_file = output_dir / "responses.jsonl"
    #with open(merged_file, 'w', encoding='utf-8') as f:
    #    for result in all_results:
    #        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return all_results


def evaluate_merged_results(output_dir, args, all_results=None):
    """

    Args:
        output_dir:
    Returns:
        results:
    """
    #if all_results is None:
    #    print("Get results from responses.jsonl!")
    #    all_results = []
    #    file_path = Path(output_dir) / "responses.jsonl"
    #    with open(file_path, 'r', encoding='utf-8') as f:
    #        for line in f:
    #            result = json.loads(line.strip())
    #            all_results.append(result)

    # read results
    predictions = []
    references = []
    problems = []
    for result in all_results:
        predictions.append(result["response"])
        references.append(result["reference"])
        if args.eval_form =="gen_code":
            problems.append(result["problem"])

    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions count ({len(predictions)}) != references count ({len(references)})"
        )

    if_correct = None

    if args.eval_form == "gen_choice":
        if_correct = get_classification_results(references, predictions, mode=args.eval_mode)
        total = len(if_correct)
        correct = 0
        for c in if_correct:
            correct += c
        accuracy = correct/total
        metric_result = {"accuracy": accuracy}

        logger.info("\n=== Eval Results ===")
        logger.info(f"  accuracy: {metric_result['accuracy']:.4f}")
    elif args.eval_form == "gen_math":
        if_correct = get_math_results(references, predictions)
        total = len(if_correct)
        correct = 0
        for c in if_correct:
            correct += c
        accuracy = correct / total
        metric_result = {"accuracy": accuracy}

        logger.info("\n=== Eval Results ===")
        logger.info(f"  accuracy: {metric_result['accuracy']:.4f}")
    elif args.eval_form == "gen_code":
        run_results = get_coding_results(references, predictions, problems)
        base_pass_at_k = calculate_coding_eval_res(run_results)
        metric_result = {"base_pass_at_k": base_pass_at_k}
        logger.info("\n=== Eval Results ===")
        logger.info(f"base_pass_at_k: {base_pass_at_k}")
    else:
        # form = gen
        rl_score = rl_eval(references, predictions)
        b2_score = b2_eval(references, predictions)
        d1_score = distinct_n(predictions, 1)
        d2_score = distinct_n(predictions, 2)
        z_coeff = zipf_coefficient(predictions)
        metric_result = {"rouge_l": rl_score, "bleu_2": b2_score, "distinct_1": d1_score, "distinct_2": d2_score, "z_coeff": z_coeff}

        logger.info("\n=== Eval Results ===")
        logger.info(f"  Rouge-L: {metric_result['rouge_l']:.4f}")
        logger.info(f"  Bleu-2: {metric_result['bleu_2']:.4f}")
        logger.info(f"  Distinct-1: {metric_result['distinct_1']:.4f}")
        logger.info(f"  Distinct-2: {metric_result['distinct_2']:.4f}")
        logger.info(f"  Zipf-coeff: {metric_result['z_coeff']:.4f}")

    merged_file = output_dir / "responses.jsonl"
    with open(merged_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(all_results):
            if if_correct is not None:
                result['result'] = if_correct[i]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return metric_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--data_path", type=str,
                        default="/path/to/data.jsonl",
                        help="Path to the test data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for inference")
    parser.add_argument("--eval_form", type=str, default="gen",
                        help="evaluation based on gen, gen_choice, gen_math, gen_code, or PPL_choice.")
    parser.add_argument("--eval_mode", type=str, default="include",
                        help="evaluation mode for gen_choice: include, case_include, strict, case_strict.")
    parser.add_argument("--sample_format", type=str, default="qa_style",
                        help="format of sample with style.")
    parser.add_argument("--style_prompt_type", type=str, default="None",
                        help="style type in prompt.")
    parser.add_argument("--style_expression_type", type=str, default="None",
                        help="style type in expression.")
    parser.add_argument("--style_domain", type=str, default="all",
                        help="style domain.")
    parser.add_argument("--model_type", type=int, default=1,
                        help="0: normal LLM; 1: layer-wise hypernetwork; 2: shared hypernetwork")
    args = parser.parse_args()

    # Set output folder
    if args.output_dir is None:
        #model_name = os.path.basename(args.model_path)
        model_name = Path(args.model_path).name
        args.output_dir = f"/path/to/metaSwiglu/eval/{model_name}_style-{args.style_domain}-{args.style_prompt_type}-{args.style_expression_type}"

    # Create output folder
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Begin Multi-GPU parallel evaluation...")
    logger.info(f"Num of GPU used: {args.num_gpus}")
    logger.info(f"Output path: {args.output_dir}")

    # Set multiprocessing method
    multiprocessing.set_start_method('spawn', force=True)

    # Start the working process
    with Pool(args.num_gpus) as pool:
        pool.map(partial(inference_worker,
                         data_path=args.data_path,
                         model_path=args.model_path,
                         output_dir=args.output_dir,
                         args=args,
                         max_samples=None if args.max_samples < 0 else args.max_samples,
                         max_new_tokens=args.max_new_tokens,
                         total_workers=args.num_gpus,
                         style_prompt_type=args.style_prompt_type,
                         style_expression_type=args.style_expression_type),
                 range(args.num_gpus))

    # merge result
    logger.info("Merge results of all working processes...")
    all_results = merge_results(args.output_dir, args.num_gpus)

    # evaluate result
    logger.info("Evaluate response quality...")
    metric_results = evaluate_merged_results(args.output_dir, args, all_results=all_results)

    # save result
    result_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metric_results, f, indent=2)

    logger.info(f"\nEvaluation results save to: {result_path}")


if __name__ == "__main__":
    main()