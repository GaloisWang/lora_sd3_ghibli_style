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

def enhanced_lora_test(official_model_path, lora_model_dir, test_prompt="a cat"):
    """增强的LoRA测试，包含多种检测方法"""
    
    print("🔄 加载基础模型...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        official_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📊 基础模型加载完成，检查初始状态...")
    
    # 记录原始权重用于对比
    original_weights = {}
    for name, param in pipe.transformer.named_parameters():
        if any(target in name for target in ["to_q", "to_k", "to_v", "to_out"]):
            original_weights[name] = param.clone().detach()
    
    print(f"记录了 {len(original_weights)} 个原始权重用于对比")
    
    print(f"🔄 加载LoRA权重: {lora_model_dir}")
    try:
        pipe.load_lora_weights(lora_model_dir)
        print("✅ LoRA权重加载命令执行成功")
    except Exception as e:
        print(f"❌ LoRA加载失败: {str(e)}")
        return False


if __name__ == "__main__":
    config = {
        'official_model_path': "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/",
        'lora_model_dir': "/home/fixed_sd3_lorav2/"
    }
    
    # 运行验证
    is_loaded = enhanced_lora_test(
        config['official_model_path'],
        config['lora_model_dir']
    )