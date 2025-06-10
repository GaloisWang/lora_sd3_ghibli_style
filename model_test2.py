from diffusers import StableDiffusion3Pipeline
from omegaconf import OmegaConf
from dataclasses import dataclass
import torch

@dataclass
class TrainParametersConfig:
    image_dir: str
    captions_dir: str
    output_dir: str
    epochs: int = 200
    learning_rate: float = 1e-4
    train_batch_size: int = 8
    rank: int = 8
    lora_alpha: int = 16
    pretrained_model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resolution: int = 512

# 假设 train_configs.pretrained_model_path 已经定义
# pretrained_model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
# 或者你的本地路径
train_parameters_config_path = "./lora_sd3_config.json" 
data_dict = OmegaConf.load(train_parameters_config_path)
train_configs = TrainParametersConfig(**data_dict)

try:
    mmdit = StableDiffusion3Pipeline.from_pretrained(
        train_configs.pretrained_model_path, use_safetensors=True, torch_dtype=torch.float16
    ).transformer

    print("--- MMDiT (Transformer) Config ---")
    print("Full MMDiT Config:")
    for key, value in mmdit.config.items():
        print(f"  {key}: {value}")

    # 替换下面的 'hidden_size' 和 'pooled_projection_dim'
    # 使用你从上面打印出来的实际键名
    # 假设通过打印发现它们可能是 'model_dim', 'clip_pooled_output_dim' 等
    # print(f"Transformer hidden_size: {mmdit.config['model_dim']}") # 替换为实际键名
    # print(f"Transformer pooled_projection_dim (input from textenc2): {mmdit.config['clip_pooled_output_dim']}") # 替换为实际键名

    # ... (其他检查代码暂时注释掉，等确认了键名再恢复) ...

except Exception as e:
    print(f"加载模型以检查配置时出错: {e}")