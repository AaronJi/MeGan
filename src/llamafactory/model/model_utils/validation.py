import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from meta_swiglu_modeling_llama import LlamaForCausalLM
# 配置参数
MODEL_PATH = "/vepfs/DI/beijing-public/models/Llama-2-7b-hf"  # 替换为实际的模型路径
PROMPT = "今天天气真好，"  # 测试用提示词
SEED = 42
MAX_NEW_TOKENS = 50

# 设置确定性生成（关闭所有随机性）
# def set_deterministic():
#     torch.manual_seed(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# 加载模型和tokenizer的通用函数
def load_model_and_tokenizer(model_class):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto")
    model = model_class.from_pretrained(MODEL_PATH, device_map="auto")
    return tokenizer, model

# 生成文本的通用函数
def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,           # 关闭采样
        # temperature=0.0,          # 温度设为0
        top_p=1.0,                # 关闭nucleus采样
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # set_deterministic()
    
    # 使用AutoModel加载
    auto_tokenizer, auto_model = load_model_and_tokenizer(AutoModelForCausalLM)
    auto_output = generate_text(auto_model, auto_tokenizer, PROMPT)
    
    # 使用LlamaForCausalLM加载
    llama_tokenizer, llama_model = load_model_and_tokenizer(LlamaForCausalLM)
    llama_output = generate_text(llama_model, llama_tokenizer, PROMPT)
    
    # 比较结果
    print("="*40 + " AutoModel Output " + "="*40)
    print(auto_output)
    
    print("\n" + "="*40 + " LlamaForCausalLM Output " + "="*40)
    print(llama_output)
    
    print("\n" + "="*40 + " Comparison " + "="*40)
    print(f"Outputs are {'IDENTICAL' if auto_output == llama_output else 'DIFFERENT'}")