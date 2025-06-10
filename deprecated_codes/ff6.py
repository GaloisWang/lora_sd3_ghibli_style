import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image, ImageChops
import os

# ===== 路径配置 =====
official_model_path = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
# lora_model_dir = "/home/models/erikhsos-campusbier-sd3-lora/"
lora_model_dir = "/home/models/MohamedRashad--sd3-civitai-lora//"
output_dir = "/root/Codes/lora_diffusion/"
os.makedirs(output_dir, exist_ok=True)

# ===== 加载 base model =====
pipe = StableDiffusion3Pipeline.from_pretrained(
    official_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.set_progress_bar_config(disable=True)

# 加载前参数
original_keys = set(pipe.transformer.state_dict().keys())

# 加载 LoRA
pipe.load_lora_weights(lora_model_dir)

# 加载后参数
new_keys = set(pipe.transformer.state_dict().keys())

# 查找是否有新增的 lora 参数
lora_keys = [k for k in new_keys - original_keys if 'lora' in k.lower()]
if lora_keys:
    print("✅ LoRA weights successfully loaded:")
    for k in lora_keys:
        print("  -", k)
else:
    print("❌ No LoRA keys found. Load may have failed.")



