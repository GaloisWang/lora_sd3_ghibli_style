import os
import gc
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from contextlib import contextmanager
from typing import Optional, List, Tuple

import os
import json
from safetensors import safe_open



def verify_lora_layers(official_model_path, lora_model_dir):
    """通过检查LoRA层存在性验证加载"""
    # 1. 加载模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        official_model_path,
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False
    ).to("cpu")
    
    # 2. 加载LoRA权重
    pipe.load_lora_weights(lora_model_dir)
    
    # 3. 检查是否存在LoRA特定层
    lora_found = False
    for name, module in pipe.transformer.named_modules():
        if "lora" in name.lower():
            print(f"🔍 检测到LoRA层: {name}")
            lora_found = True
            # 检查权重是否非零
            for param in module.parameters():
                if param.abs().sum().item() > 1e-6:
                    print(f"  权重验证: 非零值存在 ({param.abs().sum().item():.6f})")
                    return True
    
    if not lora_found:
        print("❌ 未检测到任何LoRA层")
        return False
    return True

if __name__ == "__main__":
    config = {
        'official_model_path': "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/",
        'lora_model_dir': "/home/other_peoples/"
    }
    
    # 运行验证
    is_loaded = verify_lora_layers(
        config['official_model_path'],
        config['lora_model_dir']
    )