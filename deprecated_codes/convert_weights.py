#!/usr/bin/env python3
"""
将PEFT训练的LoRA权重转换为diffusers兼容格式
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import PeftModel
import safetensors.torch as sf

def convert_peft_lora_to_diffusers(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    将PEFT LoRA权重转换为diffusers格式
    
    Args:
        base_model_path: 基础SD3模型路径
        lora_path: PEFT LoRA权重路径
        output_path: 输出路径
        device: 设备
    """
    
    print(f"加载基础模型: {base_model_path}")
    # 加载基础SD3模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
    )
    
    print(f"加载LoRA权重: {lora_path}")
    # 检查LoRA文件格式
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        # 使用safetensors格式
        lora_state_dict = sf.load_file(os.path.join(lora_path, "adapter_model.safetensors"))
    elif os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
        # 使用pytorch格式
        lora_state_dict = torch.load(os.path.join(lora_path, "adapter_model.bin"), map_location="cpu")
    else:
        raise FileNotFoundError("未找到adapter_model文件")
    
    # 转换LoRA权重格式
    print("转换LoRA权重格式...")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存为diffusers兼容格式
    print(f"保存到: {output_path}")
    
    # 方法1: 直接保存为safetensors格式（推荐）
    output_file = os.path.join(output_path, "pytorch_lora_weights.safetensors")
    sf.save_file(lora_state_dict, output_file)
    
    # 创建配置文件
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_path,
        "task_type": "DIFFUSION"
    }
    
    import json
    with open(os.path.join(output_path, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("转换完成！")
    return output_path

def load_and_test_converted_lora(base_model_path: str, lora_path: str):
    """
    测试转换后的LoRA权重
    """
    print("测试转换后的LoRA权重...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载基础模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    # 加载LoRA权重
    pipe.load_lora_weights(lora_path)
    
    # 测试生成
    prompt = "a beautiful landscape"
    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.0).images[0]
    
    # 保存测试图片
    test_output = os.path.join(lora_path, "test_output.png")
    image.save(test_output)
    print(f"测试图片已保存到: {test_output}")

if __name__ == "__main__":
    # 设置路径
    BASE_MODEL_PATH = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    LORA_PATH = "/home/lora_sd3_train_logs_fulldata_052516e5/lora_final/"
    OUTPUT_PATH = "/home/converted_lora_weights2/"
    
    try:
        # 转换LoRA权重
        # converted_path = convert_peft_lora_to_diffusers(
        #     base_model_path=BASE_MODEL_PATH,
        #     lora_path=LORA_PATH,
        #     output_path=OUTPUT_PATH
        # )

        # print("converted_path:{0}".format(converted_path))

        
        # # 测试转换结果
        load_and_test_converted_lora(BASE_MODEL_PATH, OUTPUT_PATH)
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()