#!/usr/bin/env python3
"""
基于你的训练脚本,将PEFT训练的SD3 Transformer LoRA权重转换为diffusers兼容格式
"""

import os
import torch
import json
import safetensors.torch as sf
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from peft import PeftModel, LoraConfig
import gc

def analyze_peft_lora_structure(lora_path: str):
    """
    分析PEFT LoRA文件结构
    """
    print("=== 分析PEFT LoRA结构 ===")
    
    # 检查文件存在性
    config_file = os.path.join(lora_path, "adapter_config.json")
    weight_file_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
    weight_file_bin = os.path.join(lora_path, "adapter_model.bin")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"未找到配置文件: {config_file}")
    
    weight_file = None
    if os.path.exists(weight_file_safetensors):
        weight_file = weight_file_safetensors
    elif os.path.exists(weight_file_bin):
        weight_file = weight_file_bin
    else:
        raise FileNotFoundError("未找到LoRA权重文件")
    
    print(f"✅ 配置文件: {config_file}")
    print(f"✅ 权重文件: {weight_file}")
    
    # 读取配置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"✅ LoRA配置:")
    print(f"   Rank (r): {config.get('r')}")
    print(f"   Alpha: {config.get('lora_alpha')}")
    print(f"   Target modules: {config.get('target_modules')}")
    print(f"   Task type: {config.get('task_type')}")
    
    # 读取权重
    if weight_file.endswith('.safetensors'):
        weights = sf.load_file(weight_file)
    else:
        weights = torch.load(weight_file, map_location='cpu')
    
    print(f"✅ 权重统计: {len(weights)} 个参数")
    
    # 分析权重键名模式
    sample_keys = list(weights.keys())[:10]
    print(f"权重键名示例:")
    for key in sample_keys:
        print(f"   {key}")
    
    return config, weights, weight_file

def convert_sd3_peft_to_diffusers(
    base_model_path: str,
    peft_lora_path: str,
    output_path: str
):
    """
    将你的PEFT LoRA转换为diffusers格式
    """
    print("=== SD3 PEFT LoRA 转换为 Diffusers 格式 ===")
    
    # 1. 分析PEFT结构
    peft_config, peft_weights, weight_file = analyze_peft_lora_structure(peft_lora_path)
    
    # 2. 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 3. 转换权重格式
    print("转换权重格式...")
    
    converted_weights = {}
    
    # 根据你的训练配置，目标模块是: ["to_q", "to_k", "to_v", "to_out.0"]
    # PEFT保存的权重键名通常格式为: base_model.model.{原始键名}.lora_A.weight 等
    
    for key, value in peft_weights.items():
        # 移除PEFT的包装前缀
        new_key = key
        
        # 常见的PEFT前缀
        prefixes_to_remove = [
            "base_model.model.",
            "model.",
        ]
        
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        
        converted_weights[new_key] = value
    
    print(f"✅ 转换完成: {len(converted_weights)} 个参数")
    
    # 4. 保存为diffusers格式
    output_weight_file = os.path.join(output_path, "pytorch_lora_weights.safetensors")
    sf.save_file(converted_weights, output_weight_file)
    
    # 5. 创建diffusers兼容的配置文件
    # 基于你的训练配置创建
    diffusers_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_path,
        "task_type": "DIFFUSION",
        "inference_mode": True,
        "r": peft_config.get("r", 16),
        "lora_alpha": peft_config.get("lora_alpha", 16),
        "lora_dropout": peft_config.get("lora_dropout", 0.0),
        "target_modules": peft_config.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        "bias": peft_config.get("bias", "none"),
        "fan_in_fan_out": peft_config.get("fan_in_fan_out", False),
        "init_lora_weights": peft_config.get("init_lora_weights", "gaussian")
    }
    
    config_output_path = os.path.join(output_path, "adapter_config.json")
    with open(config_output_path, 'w') as f:
        json.dump(diffusers_config, f, indent=2)
    
    print(f"✅ 转换完成!")
    print(f"   输出目录: {output_path}")
    print(f"   权重文件: {output_weight_file}")
    print(f"   配置文件: {config_output_path}")
    
    return output_path

def main():
    """
    主函数
    """
    # 设置路径 - 根据你的实际路径修改
    BASE_MODEL_PATH = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    PEFT_LORA_PATH = "/home/lora_sd3_train_logs_0602_myscript_rank16e5/lora_final/"
    OUTPUT_PATH ="/home/lora_sd3_train_logs_0602_myscript_rank16e5/lora_final_2/"
    
    try:
        # 检查输入路径
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(f"基础模型路径不存在: {BASE_MODEL_PATH}")
        
        if not os.path.exists(PEFT_LORA_PATH):
            raise FileNotFoundError(f"PEFT LoRA路径不存在: {PEFT_LORA_PATH}")
        
        # 执行转换
        converted_path = convert_sd3_peft_to_diffusers(
            base_model_path=BASE_MODEL_PATH,
            peft_lora_path=PEFT_LORA_PATH,
            output_path=OUTPUT_PATH
        )
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()