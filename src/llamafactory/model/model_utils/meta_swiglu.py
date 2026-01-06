import torch
import torch.nn as nn
from collections import OrderedDict


class MetaSwiGLUSimpleActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        # Initialize beta to a parameter tensor of the same size as intermediate_size
        self.beta = nn.Parameter(torch.ones(self.intermediate_size))

    def forward(self, input):
        # input shape = (batch_size, seq_length, intermediate_size)
        beta_unsqueezed = self.beta.unsqueeze(0).unsqueeze(0)
        return input * torch.sigmoid(beta_unsqueezed * input)


class MetaSwiGLUDataDrivenActivation0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.beta_generator = nn.Linear(config.intermediate_size, config.intermediate_size)

    def forward(self, input, style):
        # input shape = (batch_size, seq_length, intermediate_size)
        # style shape = (batch_size, seq_length, intermediate_size)
        # 将范围调整到(-0.5, 0.5)
        beta = torch.sigmoid(self.beta_generator(style)) - 0.5
        return input * torch.sigmoid((1.0 + beta) * input)

def linear_weights_zero_init(m):
    nn.init.normal_(m.weight.data, 0.0, 0.001)
    nn.init.constant_(m.bias.data, 0)

#netG.apply(weights_init)

class MetaSwiGLUDataDrivenActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.beta_generator = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.beta_generator.apply(linear_weights_zero_init)

    def forward(self, input, style=None):
        # input shape = (batch_size, seq_length, intermediate_size)
        # style shape = (batch_size, seq_length, intermediate_size)
        
        # 如果没有提供style，使用input作为style
        style = input if style is None else style
        
        # 生成beta并应用激活
        beta = torch.sigmoid(self.beta_generator(style)) - 0.5
        return input * torch.sigmoid((1.0 + beta) * input)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, default_kwargs = content if isinstance(content, tuple) else (content, {})
        return lambda **kwargs: cls(**{**default_kwargs, **kwargs})

class MetaLlamaMLP(nn.Module):
    def __init__(self, config,finetuning_args):
        super().__init__()
        self.finetuning_args = finetuning_args
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        if self.finetuning_args.meta_hidden_act == "meta_swiglu_simple":
            self.meta_act_fn = MetaSwiGLUSimpleActivation(self.config)
        elif self.finetuning_args.meta_hidden_act == "meta_swiglu":
            self.meta_act_fn = MetaSwiGLUDataDrivenActivation(self.config)

    def forward(self, x, style=None):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        
        if isinstance(self.meta_act_fn, MetaSwiGLUDataDrivenActivation):
            # 如果没有提供style，使用gate_out作为style
            style = gate_out if style is None else style
            activated = self.meta_act_fn(gate_out, style)
        else:
            activated = self.meta_act_fn(gate_out)
            
        return self.down_proj(activated * up_out)