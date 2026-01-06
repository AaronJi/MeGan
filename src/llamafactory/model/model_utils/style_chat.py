import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from meta_swiglu_modeling_llama import LlamaForCausalLM
from safetensors.torch import load_file
import gc

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
        print(f"Loading {filename}...")
        
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

def compare_beta_weights(loaded_beta_weights, model):
    """
    比较加载的beta_generator权重和模型中的beta_generator权重
    
    Args:
        loaded_beta_weights: 从checkpoint加载的beta_generator权重
        model: 已加载权重的模型
    """
    print("\n=== Beta Generator 权重比较 ===")
    
    model_state_dict = model.state_dict()
    
    for name, loaded_weight in loaded_beta_weights.items():
        if name in model_state_dict:
            # 将模型权重移到与加载权重相同的设备
            model_weight = model_state_dict[name].to(loaded_weight.device)
            
            # 检查权重是否相等
            if torch.allclose(loaded_weight, model_weight):
                status = "✅ 相同"
            else:
                # 计算差异值
                diff = torch.norm(loaded_weight - model_weight).item()
                status = f"❌ 不同 (差异: {diff:.6f})"
            
            print(f"{name}: {status}")
        else:
            print(f"{name}: ❌ 模型中不存在此权重")

def generate_style_matrix(seq_length, style_value, hidden_size=4096):
    """根据style_value生成风格矩阵"""
    return torch.full((1, seq_length, hidden_size), style_value)

def main():
    print("开始加载模型...")
    
    # 模型路径
    base_model_path = "/vepfs/DI/beijing-public/models/Meta-Llama-3-8B"
    model_path = "/vepfs/DI/user/xiningyuan/results/metaSwiglu/metaSwiGLU_llama3_8b_cpt_Shakespeare_tanh_0825"
    
    # 初始化标准模型（原版Llama）
    print("加载标准模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 生成标准回复
    print("\n生成标准回复...")
    base_question = "I think he"
    inputs = base_tokenizer(base_question, return_tensors="pt").to(base_model.device)
    
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=base_tokenizer.eos_token_id
        )
    
    base_response = base_tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"标准模型回复: {base_response}")
    
    # 清理标准模型
    del base_model
    del base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # 初始化自定义模型（meta-SwiGLU）
    print("\n加载自定义模型...")
    config = transformers.AutoConfig.from_pretrained(model_path)
    custom_model = LlamaForCausalLM(config)
    
    # 加载自定义权重
    _ = load_safetensors_checkpoint(custom_model, model_path)
    custom_model = custom_model.to(device='cuda', dtype=torch.bfloat16)
    custom_model.eval()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 生成不同风格回复
    print("\n生成不同风格回复...")
    # question = "The meeting was incredibly"
    question = base_question
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_length = input_ids.size(1)
    
    # for style_value in ['loyalty', 'care', 'fairness']:
    for style_value in ['modern', 'original']:
        # 生成风格矩阵
        # style_matrix = generate_style_matrix(seq_length, style_value)
        style_ids = tokenizer(style_value, return_tensors="pt")['input_ids'].to(device='cuda')
        # input_embeddings = custom_model.get_input_embeddings().to(device='cuda')
        # style_matrix = input_embeddings(style_ids)
        # style_matrix = style_matrix.to(device='cuda', dtype=torch.bfloat16)
        
        # 准备输入
        input_ids_cuda = input_ids.to(custom_model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = custom_model.generate(
                input_ids_cuda,
                style=style_ids,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id 
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n风格值 beta={style_value}:")
        print(f"自定义模型回复: {response}")

if __name__ == "__main__":
    main()