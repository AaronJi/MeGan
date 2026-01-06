import os
import json
import torch
import transformers
from transformers import AutoTokenizer
from meta_swiglu_modeling_llama import LlamaForCausalLM
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保nltk资源已下载
# nltk.download('punkt', quiet=True)

def load_safetensors_checkpoint(model, checkpoint_path):
    """
    从分片的safetensors文件加载模型权重，并返回beta_generator权重
    
    Args:
        model: 初始化的模型
        checkpoint_path: 模型文件夹路径
    
    Returns:
        loaded_beta_weights: 从checkpoint加载的beta_generator权重
    """
    # 获取所有safetensors文件
    safetensors_files = sorted([
        f for f in os.listdir(checkpoint_path) 
        if f.endswith('.safetensors')
    ])
    
    # 用于存储从checkpoint加载的beta_generator权重
    loaded_beta_weights = {}
    
    # 遍历所有分片文件
    for filename in safetensors_files:
        file_path = os.path.join(checkpoint_path, filename)
        logger.info(f"Loading {filename}...")
        
        # 加载当前分片的权重
        state_dict = load_file(file_path)
        
        # 提取beta_generator权重用于比较
        for name, param in state_dict.items():
            if "beta_generator" in name:
                loaded_beta_weights[name] = param.clone()
        
        # 将权重加载到模型中
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的键
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and model_dict[k].shape == v.shape
        }
        
        # 更新模型权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    
    return loaded_beta_weights

def rl_eval(ground_truth: list, predictions: list):
    """
    计算Rouge-L分数
    
    Args:
        ground_truth: 参考文本列表
        predictions: 生成文本列表
    
    Returns:
        rougeL_fmeasure: Rouge-L F1分数
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("References and predictions must have the same length")
    
    # 使用rouge_score库替代原来的rouge库
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for ref, pred in zip(ground_truth, predictions):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)
    
    return sum(scores) / len(scores)

def b2_eval(ground_truth:list, predictions:list):
    if len(ground_truth)!=len(predictions):
        raise ValueError("Sentence bleu requires the same number of references for each prediction")

    ground_truth_list = [tmp.split(' ') for tmp in ground_truth]
    predictions_list = [tmp.split(' ') for tmp in predictions]
    b2_res_list = [sentence_bleu([reference], hypothesis, weights = [0.5,0.5], smoothing_function = SmoothingFunction().method3) for reference,hypothesis in zip(ground_truth_list, predictions_list)]
    b2_res = sum(b2_res_list)/len(b2_res_list)
    return b2_res

def inference_worker(rank, data_path, model_path, output_dir, max_new_tokens=100, total_workers=8):
    """
    多卡推理的工作进程函数
    
    Args:
        rank: 进程排名
        data_path: 数据文件路径
        model_path: 模型路径
        output_dir: 输出目录
        max_new_tokens: 最大生成长度
        total_workers: 总工作进程数
    """
    # 设置当前进程可见的GPU
    start_gpu = rank
    gpu_list = [start_gpu]
    gpu_list_str = ','.join(map(str, gpu_list))
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str
    device = torch.device(f'cuda:{rank}')
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'responses_part_{rank}.jsonl'
    
    # 加载数据集
    questions, references, styles = load_dataset(data_path)
    
   # 分割数据
    data_tuples = list(zip(questions, styles, references, range(len(questions))))
    data_slice = np.array_split(data_tuples, total_workers)[rank]

    # 加载现有结果（如果存在）
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
    
    # 初始化模型和分词器
    logger.info(f"Worker {rank} loading model...")
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config)
    _ = load_safetensors_checkpoint(model, model_path)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side='left',
        truncation_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 处理数据 - 逐条处理
    for i, (question, style, reference, original_idx) in enumerate(tqdm(data_slice, desc=f"Worker {rank}")):
        # 检查是否已处理
        input_hash = hashlib.md5((question + style).encode()).hexdigest()
        if input_hash in existing_results:
            continue
            
        if style is None:
            # 如果没有风格，跳过
            continue
            
        try:
            # 使用apply_chat_template处理用户消息
            user_message = [{"role": "user", "content": question}]
            chat_input = tokenizer.apply_chat_template(user_message, return_tensors="pt", add_generation_prompt=True)
            
            # 编码风格文本
            style_input = tokenizer(
                style, 
                return_tensors="pt", 
                add_special_tokens=False
            )["input_ids"]
            
            # 移动到GPU
            chat_input = chat_input.to(device)
            style_input = style_input.to(device)
            
            # 单条推理
            with torch.no_grad():
                output = model.generate(
                    input_ids=chat_input,
                    style=style_input,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True
                )
            
            # 解码结果
            # 跳过输入部分，只保留生成的部分
            response = tokenizer.decode(
                output[0][len(chat_input[0]):], 
                skip_special_tokens=True
            )
            
            # 创建结果记录
            record = {
                "index": original_idx,
                "question": question,
                "style": style,
                "reference": reference,
                "response": response
            }
            
            # 立即保存到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Error processing sample {original_idx}: {e}")
            # 保存错误信息
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

def load_dataset(data_path, max_samples=None):
    """
    加载数据集
    
    Args:
        data_path: 数据集路径
        max_samples: 最大样本数（用于测试）
    
    Returns:
        questions: 问题列表
        references: 参考回复列表
        styles: 风格列表
    """
    questions = []
    references = []
    styles = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
                
            data = json.loads(line)
            questions.append(data['messages'][0]['content'])
            references.append(data['messages'][1]['content'])
            styles.append(data['messages'][2]['content'])
    
    return questions, references, styles

def merge_results(output_dir, total_workers):
    """
    合并所有工作进程的结果
    
    Args:
        output_dir: 输出目录
        total_workers: 总工作进程数
    
    Returns:
        all_results: 合并后的结果列表
    """
    output_dir = Path(output_dir)
    all_results = []
    
    for rank in range(total_workers):
        part_file = output_dir / f'responses_part_{rank}.jsonl'
        if part_file.exists():
            try:
                with open(part_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line.strip()))
            except Exception as e:
                logger.error(f"Error loading part {rank}: {e}")
    
    # 按索引排序
    all_results.sort(key=lambda x: x["index"])
    
    # 保存合并后的结果
    merged_file = output_dir / "responses.jsonl"
    with open(merged_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return all_results

def evaluate_merged_results(output_dir):
    """
    评估合并后的结果
    
    Args:
        output_dir: 输出目录
    
    Returns:
        results: 评估结果字典
    """
    file_path = Path(output_dir) / "responses.jsonl"
    
    # 读取结果
    predictions = []
    references = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions.append(data["response"])
            references.append(data["reference"])
    
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions count ({len(predictions)}) != references count ({len(references)})"
        )

    rl_score = rl_eval(references, predictions)
    b2_score = b2_eval(references, predictions)

    return {"rouge_l": rl_score, "bleu_2": b2_score}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="/vepfs/DI/beijing-public/datasets/stylized_fmt/mic_fmt/mic_test_style.jsonl",
                      help="Path to the test data")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for results")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                      help="Maximum number of new tokens to generate")
    parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of GPUs to use for inference")
    args = parser.parse_args()
    
    # 设置输出目录
    model_name = os.path.basename(args.model_path)
    args.output_dir = f"/vepfs/DI/user/xiningyuan/results/metaSwiglu_eval/{model_name}"

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("开始多卡并行评估模型...")
    logger.info(f"使用GPU数量: {args.num_gpus}")
    logger.info(f"输出路径: {args.output_dir}")
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    
    # 启动工作进程
    with Pool(args.num_gpus) as pool:
        pool.map(partial(inference_worker,
                        data_path=args.data_path,
                        model_path=args.model_path,
                        output_dir=args.output_dir,
                        max_new_tokens=args.max_new_tokens,
                        total_workers=args.num_gpus),
                 range(args.num_gpus))
    
    # 合并结果
    logger.info("合并所有工作进程的结果...")
    all_results = merge_results(args.output_dir, args.num_gpus)
    
    # 评估结果
    logger.info("评估回复质量...")
    results = evaluate_merged_results(args.output_dir)
    
    # 打印结果
    logger.info("\n=== 评估结果 ===")
    logger.info(f"  Rouge-L: {results['rouge_l']:.4f}")
    logger.info(f"  Bleu-2: {results['bleu_2']:.4f}")
    
    # 保存结果
    result_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n评估结果已保存至: {result_path}")

if __name__ == "__main__":
    main()