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


# 只加载transformer以检查
try:
    mmdit = StableDiffusion3Pipeline.from_pretrained(
        train_configs.pretrained_model_path, use_safetensors=True, torch_dtype=torch.float16
    ).transformer

    print("--- MMDiT (Transformer) Config ---")
    print(f"Transformer hidden_size: {mmdit.config.hidden_size}") # 应该是 1536
    print(f"Transformer pooled_projection_dim (input from textenc2): {mmdit.config.pooled_projection_dim}") # 应该是 1280

    print("\n--- pooled_projection_proj Layer (projects pooled_projections) ---")
    # self.pooled_projection_proj = nn.Linear(config.pooled_projection_dim, config.hidden_size)
    print(f"mmdit.pooled_projection_proj: {mmdit.pooled_projection_proj}")
    # 期望: Linear(in_features=1280, out_features=1536, bias=True)

    print("\n--- time_text_embed Layer (CombinedTimestepTextProjEmbeddings) ---")
    # self.time_text_embed = CombinedTimestepTextProjEmbeddings(
    #     config.hidden_size, # timestep_input_dim
    #     config.hidden_size, # text_embed_dim (this should be input from pooled_projection_proj's output)
    #     config.hidden_size * 4, # time_embed_dim
    # )
    # 所以 text_embed_dim 应该是 1536

    # 获取 time_text_embed 期望的 text_embed_dim (通过其子模块 linear_1 的 in_features)
    # time_text_embed -> text_embedder (PixArtAlphaTextProjection) -> linear_1
    time_text_embed_text_embed_dim_expected_by_linear1 = mmdit.time_text_embed.text_embedder.linear_1.in_features
    print(f"mmdit.time_text_embed.text_embedder.linear_1 in_features: {time_text_embed_text_embed_dim_expected_by_linear1}")
    # 根据错误是 2048, 根据标准配置应该是 1536 (即 transformer.config.hidden_size)
    print(f"mmdit.time_text_embed.text_embedder.linear_1: {mmdit.time_text_embed.text_embedder.linear_1}")


except Exception as e:
    print(f"加载模型以检查配置时出错: {e}")