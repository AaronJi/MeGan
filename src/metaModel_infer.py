
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_llama import LlamaMLP
#from src.llamafactory.model.model_utils.meta_swiglu import MetaLlamaMLP
from llamafactory.model.model_utils.meta_swiglu import MetaLlamaMLP
from llamafactory.model.patcher import (patch_activation_model)

model_name_or_path = "/user/jl/results/metaSwiglu/metaSwiGLU_llama2_7b_gsm8k_lr-3_zeroinit"

from llamafactory.hparams.finetuning_args import FinetuningArguments

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)



finetuning_args = FinetuningArguments()
finetuning_args.finetuning_type = "meta_swiglu"
finetuning_args.meta_hidden_act = "meta_swiglu"
#print(finetuning_args)
#print('&&&&&')

# only load llama structure
config = AutoConfig.from_pretrained(model_name_or_path) # model_args.model_name_or_path, **init_kwargs
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
#print(model)

patch_activation_model(model, LlamaMLP, MetaLlamaMLP, config, finetuning_args)
print("after patch")

print(model)

#exit(2)


# 输出meta llama中的参数，并输出是否可以训练
for name, module in model.named_modules():
    if isinstance(module, MetaLlamaMLP):
        print("module is MetaLlamaMLP")
        for sub_name, param in module.named_parameters():
            print(f"MetaLlamaMLP Parameter: {name}.{sub_name}, requires_grad: {param.requires_grad}")

from llamafactory.extras.misc import count_parameters
trainable_params, all_param = count_parameters(model)
is_trainable = False
if is_trainable:
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
else:
    param_stats = f"all params: {all_param:,}"
print(param_stats)


from safetensors import safe_open
from transformers.utils import cached_file

#kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}

#parts = 3
parts = 5
state_dict = {}
for part in range(parts):
    filename = f"model-0000{part+1}-of-0000{parts}.safetensors"
    print(filename)

    tensor_file = cached_file(filename=filename, path_or_repo_id=model_name_or_path)
    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        state_dict_ = {key: f.get_tensor(key) for key in f.keys()}
    state_dict.update(state_dict_)

#print(len(state_dict))
#print(state_dict.keys())
#exit(2)

model.load_state_dict(state_dict, strict=True)

#from safetensors.torch import load_file

model.eval()
model.requires_grad_(False)

# <s>  
prompt = "[INST]<<SYS>>" + "\n"
prompt += "You are a helpful assistant." + "\n"
prompt += "<</SYS>>" + "\n" + "\n"
prompt += "Please answer the following question with true or false, question: do iran and afghanistan speak the same language?" + "\n"
prompt += "Answer format: true/false [/INST]"  #  the correct answer is true</s>
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt")
#response = model.generate([prompt], 512)[0]
print(inputs)

# 执行推理
outputs = model.generate(inputs.input_ids, max_new_tokens=16)
print(outputs)
# 解码输出结果
response = tokenizer.decode(outputs[0], skip_special_tokens=False, eos_token_id=2) # "</s>"
print('***')
print(response)

