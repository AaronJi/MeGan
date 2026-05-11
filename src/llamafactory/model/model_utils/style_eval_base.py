#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练语言模型评估脚本（续写 + PPL + 额外无监督指标）
"""
import argparse
import json
import logging
import math
import os
from collections import Counter
from functools import partial
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from safetensors.torch import load_file
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from typing import List, Tuple

from meta_swiglu_modeling_llama import LlamaForCausalLM   # 你的自定义 Llama

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. 通用工具
# ------------------------------------------------------------------
def load_safetensors_checkpoint(model, checkpoint_path):
    safetensors_files = sorted([
        f for f in os.listdir(checkpoint_path)
        if f.endswith('.safetensors')
    ])
    for filename in safetensors_files:
        file_path = os.path.join(checkpoint_path, filename)
        logger.info(f"Loading {filename}...")
        state_dict = load_file(file_path)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


# ------------------------------------------------------------------
# 2. 评估指标
# ------------------------------------------------------------------
def compute_ppl(model, tokenizer, full, style_ids, device):
    """
    逐句计算 PPL，避免 full 为原文列表。
    """
    model.eval()
    ppls = []
    with torch.no_grad():
        # 单独编码每个句子
        encodings = tokenizer(
            full,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            padding=False  # 不进行填充
        ).to(device)
        
        input_ids = encodings.input_ids
        # 对于因果语言模型，标签是向右偏移一位的输入
        target_ids = input_ids.clone()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # 前向传播
        outputs = model(**encodings, style=style_ids)
        logits = outputs.logits
        
        # 预测分布向左偏移一位，与标签对齐
        shift_logits = logits[..., :-1, :].contiguous()
        
        # 计算每个标记的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        
        # 计算有效标记的数量（序列长度减1，因为第一个标记没有预测）
        num_valid_tokens = shift_labels.size(1)
        
        # 对每个句子的损失求和并除以有效标记数
        sum_loss = loss.sum()
        
        if num_valid_tokens > 0:
            ppl = torch.exp(sum_loss / num_valid_tokens).item()
        else:
            ppl = float('inf')  # 处理空序列的情况
    return ppl


def rl_score(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure
              for ref, pred in zip(references, predictions)]
    return sum(scores)/len(scores)


def bleu2_score(references, predictions):
    weights = [0.5, 0.5]
    smoothie = SmoothingFunction().method3
    refs = [r.split() for r in references]
    hyps = [p.split() for p in predictions]
    scores = [sentence_bleu([r], h, weights=weights, smoothing_function=smoothie)
              for r, h in zip(refs, hyps)]
    return sum(scores)/len(scores)


# ---------- 额外无监督指标 ----------
def distinct_n(seqs, n=2):
    grams = [g for s in seqs for g in zip(*[s[i:] for i in range(n)])]
    return len(set(grams)) / max(len(grams), 1)


def zipf_coefficient(seqs):
    """
    拟合 log(rank) ~ log(freq) 的斜率作为 Zipf 系数
    越接近 -1 越自然
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


# ------------------------------------------------------------------
# 3. 数据读取（适配新格式）
# ------------------------------------------------------------------
def load_dataset(data_path: str, max_samples: int = None) -> Tuple[List[str], List[str], List[str]]:
    """
    兼容两种格式：
      - *.json   -> 顶层为 list，每个元素一条样本
      - *.jsonl  -> 每行一个 json dict
    返回:
        prefixes   : 用于续写的前缀（目前与 full_texts 相同，可后续截断）
        full_texts : 完整原文，用于 PPL 与续写评估
        styles     : 风格标签
    """
    prefixes, full_texts, styles = [], [], []

    # 根据扩展名决定读取方式
    if str(data_path).lower().endswith(".jsonl"):
        with open(data_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and idx >= max_samples:
                    break
                item = json.loads(line.strip())
                user_turn  = item["messages"][0]["content"]
                style_turn = item["messages"][1]["content"]
                prefixes.append(user_turn)
                full_texts.append(user_turn)
                styles.append(style_turn)

    else:  # .json
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)  # list of samples
            if max_samples is not None:
                data = data[:max_samples]
            for item in data:
                user_turn  = item["messages"][0]["content"]
                style_turn = item["messages"][1]["content"]
                prefixes.append(user_turn)
                full_texts.append(user_turn)
                styles.append(style_turn)

    return prefixes, full_texts, styles


# ------------------------------------------------------------------
# 4. 多卡工作进程（续写 + PPL）
# ------------------------------------------------------------------
def inference_worker(
    rank, data_path, model_path, output_dir,
    prefix_ratio=2, max_new_tokens=128,
    total_workers=8
):
    """
    每个 GPU 进程：
      1. 计算自己 slice 的 PPL
      2. 用 prefix_len 个 token 做续写，计算 Rouge-L / BLEU-2
    """
    device = torch.device(f'cuda:{rank}')
    os.makedirs(output_dir, exist_ok=True)
    out_file = Path(output_dir) / f'part_{rank}.jsonl'

    # 加载模型 & tokenizer
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config)
    _ = load_safetensors_checkpoint(model, model_path)
    model.to(device).eval()#.half()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 读取数据
    prefixes, full_texts, styles = load_dataset(data_path)
    slice_data = np.array_split(list(zip(prefixes, full_texts, styles)), total_workers)[rank]

    # 续写与 PPL 计算
    ppls = []
    preds, refs = [], []

    for prefix, full, style in tqdm(slice_data, desc=f'GPU{rank}'):
        # ---------- 续写 ----------
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_len = int(len(prompt_ids) / prefix_ratio)
        input_ids = torch.tensor(prompt_ids[:prefix_len], device=device).unsqueeze(0).to(device)
        # print(style)
        style_ids = tokenizer(
                style, 
                return_tensors="pt", 
                add_special_tokens=False
        )["input_ids"].to(device)
        # print(style_ids)
        # 单条推理
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                style=style_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True
            )
            gen_text = tokenizer.decode(gen_ids[0, prefix_len:], skip_special_tokens=True)
        #print(input_ids.cpu().numpy().tolist())
        #print(type(input_ids.cpu().numpy().tolist()))
        prefix_text = tokenizer.decode(input_ids[0,:].cpu().numpy().tolist(), skip_special_tokens=True) # concat the prefix text back
        preds.append(prefix_text + gen_text)
        refs.append(full)

        # ---------- PPL ----------
        ppl = compute_ppl(model, tokenizer, full, style_ids, device)
        ppls.append(ppl)

        # 即时保存
        rec = {
            "prefix": prefix,  # 近似字符长度
            "full": full,
            "generated": gen_text,
            "ppl": float(ppl),
            "style": style
        }
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    logger.info(f'GPU{rank} finished. '
                f'Avg PPL={np.mean(ppls):.2f} '
                f'R-L={rl_score(refs, preds):.4f} '
                f'BLEU-2={bleu2_score(refs, preds):.4f}')


# ------------------------------------------------------------------
# 5. 合并结果 & 计算最终指标
# ------------------------------------------------------------------
def merge_and_eval(output_dir, total_workers):
    all_records = []
    for r in range(total_workers):
        part = Path(output_dir) / f'part_{r}.jsonl'
        if part.exists():
            with open(part) as f:
                all_records.extend([json.loads(l) for l in f])
    # 写总文件
    merged = Path(output_dir) / 'all_records.jsonl'
    with open(merged, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # 计算指标
    preds = [r['generated'] for r in all_records]
    refs  = [r['full'] for r in all_records]
    ppls  = [r['ppl'] for r in all_records]

    metrics = {
        "PPL": float(np.mean(ppls)),
        "Rouge-L": rl_score(refs, preds),
        "BLEU-2": bleu2_score(refs, preds),
        "Distinct-1": distinct_n(preds, 1),
        "Distinct-2": distinct_n(preds, 2),
        "Zipf-coeff": zipf_coefficient(preds)
    }

    with open(Path(output_dir) / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("=== 最终指标 ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    return metrics


# ------------------------------------------------------------------
# 6. 主入口
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix_ratio", type=int, default=2,
                        help="续写时给模型看的前缀 token 数")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    model_name = Path(args.model_path).name
    args.output_dir = args.output_dir or f"/user/xiningyuan/results/metaSwiglu_eval/{model_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    multiprocessing.set_start_method('spawn', force=True)
    with Pool(args.num_gpus) as pool:
        pool.map(partial(inference_worker,
                         data_path=args.data_path,
                         model_path=args.model_path,
                         output_dir=args.output_dir,
                         prefix_ratio=args.prefix_ratio,
                         max_new_tokens=args.max_new_tokens,
                         total_workers=args.num_gpus),
                 range(args.num_gpus))

    merge_and_eval(args.output_dir, args.num_gpus)


if __name__ == "__main__":
    main()