import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from meta_swiglu_modeling_llama import LlamaForCausalLM
from safetensors.torch import load_file
import gc

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


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


def load_custom_model(model_path):
    print("\n加载自定义模型...")
    config = transformers.AutoConfig.from_pretrained(model_path)
    custom_model = LlamaForCausalLM(config)

    # 加载自定义权重
    _ = load_safetensors_checkpoint(custom_model, model_path)
    custom_model = custom_model.to(device='cuda', dtype=torch.bfloat16)
    custom_model.eval()
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer, custom_model


def generate(query, style, tokenizer, custom_model):
    inputs = tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]
    style_ids = tokenizer(style, return_tensors="pt")['input_ids'].to(device='cuda')
    # 准备输入
    input_ids_cuda = input_ids.to(custom_model.device)
    # 生成回复
    with torch.no_grad():
        outputs = custom_model.generate(
            input_ids_cuda,
            style=style_ids,
            max_length=1024,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 页面设置
st.set_page_config(
    page_title="AI聊天助手",
    page_icon="🤖",
    layout="wide"
)
# 标题和说明
st.title("🤖 AI聊天助手")
st.markdown("""
欢迎使用AI聊天助手！您可以通过输入模型地址来加载自定义的大语言模型，然后开始对话。
支持Hugging Face模型库中的模型或本地模型路径。
""")

# 侧边栏 - 模型配置
with st.sidebar:
    st.header("⚙️ 模型配置")

    # 模型地址输入
    model_path = st.text_input(
        "🔗 模型地址",
        placeholder="输入模型路径（Hugging Face ID或本地路径）",
        help="例如：deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 或 /path/to/your/model"
    )

    # 模型参数配置
    st.subheader("模型参数")
    max_length = st.slider("最大生成长度", 100, 2048, 512, help="控制生成文本的最大长度")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1, help="控制生成文本的随机性")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.1, help="核采样参数")


# 模型加载函数
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """加载模型和tokenizer"""
    try:
        with st.spinner(f"正在加载模型 {model_path}，请稍候..."):
            tokenizer, model = load_custom_model(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None


# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是AI助手，请先加载模型然后开始对话。"}
    ]

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "current_model_path" not in st.session_state:
    st.session_state.current_model_path = ""

# 模型加载区域
st.header("🚀 模型加载")

col1, col2 = st.columns([3, 1])
with col1:
    model_path_input = st.text_input(
        "模型路径",
        value=model_path if model_path else "",
        key="model_path_input",
        placeholder="例如：Qwen/Qwen-1_8B-Chat"
    )

with col2:
    load_clicked = st.button("加载模型", type="primary", use_container_width=True)

# 模型加载逻辑
if load_clicked and model_path_input:
    if st.session_state.model_loaded and st.session_state.current_model_path == model_path_input:
        st.info("模型已加载，无需重新加载")
    else:
        # 清除之前的模型缓存
        if st.session_state.model_loaded:
            st.cache_resource.clear()

        # 加载新模型
        tokenizer, model = load_model(model_path_input)

        if tokenizer is not None and model is not None:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.current_model_path = model_path_input
            st.session_state.messages = [
                {"role": "assistant", "content": f"模型 {model_path_input} 加载成功！请问有什么可以帮助您的？"}
            ]
            st.success(f"模型 {model_path_input} 加载成功！")
        else:
            st.session_state.model_loaded = False
            st.error("模型加载失败，请检查模型路径是否正确")

# 显示模型状态
if st.session_state.model_loaded:
    st.success(f"✅ 模型已加载: {st.session_state.current_model_path}")
else:
    st.warning("❌ 尚未加载模型，请先输入模型路径并点击加载模型")

# 聊天界面
st.header("💬 对话界面")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

style = st.text_input("请输入回复风格", placeholder="例如：formal,informal")

# 用户输入
if query := st.chat_input("请输入您的问题..."):
    # 检查模型是否已加载
    if not st.session_state.model_loaded:
        st.error("请先加载模型再开始对话！")
        st.stop()

    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": query})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)

    # 生成助手回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 准备输入
                tokenizer = st.session_state.tokenizer
                model = st.session_state.model
                with torch.no_grad():
                    response = generate(query, style, tokenizer, model)
                # 显示回复
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"生成回复时出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# 聊天控制
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("清空对话历史", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "对话历史已清空，请问有什么可以帮助您的？"}
        ]
        st.rerun()

with col2:
    if st.button("重新加载模型", use_container_width=True):
        if st.session_state.current_model_path:
            st.cache_resource.clear()
            tokenizer, model = load_model(st.session_state.current_model_path)
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.success("模型重新加载成功！")
                st.rerun()