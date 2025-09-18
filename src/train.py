import os
import torch
import torch.nn as nn
import timm
from typing import Optional

class LoraLayer(nn.Module):
    """ Injects a LoRA layer into a linear layer. """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int = 1):
        super().__init__()
        self.original_layer = original_layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.scaling = alpha / rank
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

def add_lora_to_deit(model: nn.Module, rank: int):
    """ Adds LoRA layers to the QKV projections of a DeiT model. """
    for block in model.blocks:
        qkv_layer = block.attn.qkv
        lora_qkv = LoraLayer(qkv_layer, rank=rank)
        block.attn.qkv = lora_qkv
    return model

def prepare_source_model(model_name: str, lora_rank: Optional[int], save_dir: str) -> str:
    """ 
    Prepares and saves a pre-trained source model.
    Loads from timm, adds LoRA if specified, and saves the state dict.
    """
    print(f"Preparing source model: {model_name}")
    os.makedirs(save_dir, exist_ok=True)

    model_filename = model_name.replace('/', '_')
    if lora_rank and lora_rank > 0:
        model_filename += f"_lora{lora_rank}"
    save_path = os.path.join(save_dir, f"{model_filename}.pth")

    if os.path.exists(save_path):
        print(f"Model already prepared at {save_path}")
        return save_path

    # All models are loaded from timm for consistency
    model_map = {
        'resnet50': 'resnet50.a1_in1k',
        'convnext_tiny': 'convnext_tiny.in12k_ft_in1k',
        'deit_small': 'deit_small_patch16_224.fb_in1k',
        'mobilevit_xs': 'mobilevit_xs.cvdf_in1k',
    }
    timm_model_name = model_map.get(model_name, model_name)

    try:
        model = timm.create_model(timm_model_name, pretrained=True)
    except Exception as e:
        print(f"Failed to create model {timm_model_name} from timm.")
        raise e

    if 'deit' in model_name and lora_rank and lora_rank > 0:
        print(f"Adding LoRA with rank {lora_rank} to {model_name}")
        model = add_lora_to_deit(model, rank=lora_rank)

    model.eval()

    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

    return save_path
