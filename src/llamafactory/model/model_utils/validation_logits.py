import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from meta_swiglu_modeling_llama import LlamaForCausalLM

MODEL_PATH = "/beijing-public/models/Llama-2-7b-hf"
PROMPT = "今天天气真好，"
SEED = 42
MAX_NEW_TOKENS = 50
torch.set_printoptions(precision=32)


def set_deterministic():
    """增强确定性设置"""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_class):
    """带设备同步的模型加载"""
    # 确保tokenizer和模型使用相同设备
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = model_class.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    
    # 同步设备信息
    device = model.device
    tokenizer.device = device
    return tokenizer, model

def generate_with_logits(model, tokenizer, prompt):
    """带logits输出的生成函数"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 启用logits输出
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,          # 启用logits输出
        return_dict_in_generate=True # 返回结构化结果
    )
    
    # 提取logits
    logits_sequence = [logits.cpu() for logits in outputs.scores]
    full_logits = torch.stack(logits_sequence)
    
    # 解码文本
    decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return decoded_text, full_logits

def compare_logits(logits_a, logits_b):
    """详细logits比较函数"""
    if logits_a.shape != logits_b.shape:
        print(f"形状不匹配: {logits_a.shape} vs {logits_b.shape}")
        return False
    
    # 绝对差异比较
    diff = torch.abs(logits_a - logits_b)
    max_diff = torch.max(diff).item()
    avg_diff = torch.mean(diff).item()
    
    print(f"最大绝对差异: {max_diff:.6f}")
    print(f"平均绝对差异: {avg_diff:.6f}")
    
    # 精确匹配检查
    exact_match = torch.all(logits_a == logits_b).item()
    print(f"精确匹配: {'是' if exact_match else '否'}")
    
    return exact_match

if __name__ == "__main__":
    set_deterministic()  # 恢复确定性设置
    
    # 使用AutoModel加载
    auto_tokenizer, auto_model = load_model_and_tokenizer(AutoModelForCausalLM)
    auto_output, auto_logits = generate_with_logits(auto_model, auto_tokenizer, PROMPT)
    
    # 使用LlamaForCausalLM加载
    llama_tokenizer, llama_model = load_model_and_tokenizer(LlamaForCausalLM)
    llama_output, llama_logits = generate_with_logits(llama_model, llama_tokenizer, PROMPT)
    
    # 文本比较
    print("\n" + "="*40 + " 文本比较 " + "="*40)
    print(f"输出是否相同: {'是' if auto_output == llama_output else '否'}")
    print(llama_output)
    print(auto_output)
    
    # Logits比较
    print("\n" + "="*40 + " Logits比较 " + "="*40)
    exact_match = compare_logits(auto_logits, llama_logits)
    
    # 详细logits打印（前3个token）
    print("\n" + "="*40 + " Logits示例 " + "="*40)
    for i in range(3):
        print(f"\nToken {i+1}:")
        print(f"AutoModel 最大值: {torch.max(auto_logits[i]).item():.4f}")
        print(f"LlamaModel最大值: {torch.max(llama_logits[i]).item():.4f}")
        print(f"前5个值差异: {torch.abs(auto_logits[i][:5] - llama_logits[i][:5])}")