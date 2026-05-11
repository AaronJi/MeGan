# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...hparams import DataArguments


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

    if not data_args.packing:
        if data_args.template == "gemma":
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        result = tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)
        result['style'] = []
        # 定义风格到分数的映射
        style_map = {
            "very positive": 0.75,
            "positive": 0.5,
            "neutral": 0.0,
            "negative": -0.5,
            "very negative": -0.75
        }

        for i in range(len(result["input_ids"])):
                        # 处理style文本
            if examples["_style"][i]:
                # print(examples["_style"][i])
                # 如果是positive，生成全1矩阵，否则全0矩阵
                # is_positive = examples["_style"][i].lower() in ["pos", "positive"]
                # shape: [seq_length, intermediate_size]
                # style = torch.ones(seq_length, 11008) if is_positive \
                #        else torch.zeros(seq_length, 11008)
                style = tokenizer.encode(
                    examples["_style"][i],
                    add_special_tokens=False
                )
                # print(style)
                result["style"].append(style)
            else:
                style = torch.zeros(seq_length, 11008)
                result["style"].append(style)

            # seq_length = len(result["input_ids"][i])
            # if examples["_style"][i]:
            #     style_label = examples["_style"][i].lower().strip()  # 统一格式
            #     # 获取对应分数，若无对应则默认为neutral
            #     score = style_map.get(style_label, style_map["neutral"])
            #     # 生成全分数矩阵 [seq_length, 11008]
            #     style = torch.full((seq_length, 4096), score)
            # else:
            #     # 空值处理为中性分数
            #     style = torch.full((seq_length, 4096), style_map["neutral"])
            
            # # score = (float(examples["_style"][i]) - 0.5) * 2
            # # style = torch.full((seq_length, 4096), score)
            # # print(f"{style}\n{style}")

            # result["style"].append(style)
        
    else:
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if data_args.template == "gemma":
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id

    return result
