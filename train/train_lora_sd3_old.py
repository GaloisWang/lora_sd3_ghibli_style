import copy
import glob
import os
import torch

# import torch.nn.functional as F
from PIL import Image

from dataclasses import dataclass
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

# from diffusers.training_utils import get_noise_sampler
from omegaconf import OmegaConf
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)  # 导入用于处理SD3多文本编码器的模型和分词器


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


def load_training_config(config_path: str) -> TrainParametersConfig:
    ## 从json文件中加载训练参数配置
    data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)


class Text2ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, captions_dir, pretrained_model_path, resolution):
        self.image_paths = []
        self.image_paths.extend(
            glob.glob(os.path.join(os.path.abspath(image_dir), "*.jpg"))
        )
        self.image_paths.extend(
            glob.glob(os.path.join(os.path.abspath(image_dir), "*.png"))
        )
        self.image_paths = sorted(self.image_paths)
        caption_paths = sorted(
            glob.glob(os.path.join(os.path.abspath(captions_dir), "*.txt"))
        )
        captions = []
        for caption_path in caption_paths:
            with open(caption_path, "r", encoding="utf-8") as f:
                captions.append(f.readline().strip())

        if len(captions) != len(self.image_paths):
            raise ValueError("图像数量与文本标注数量不一致，请检查数据集。")

        self.captions = captions
        self.resolution = resolution
        # 注意：这里仅用于获取 image_processor 和 tokenizer，避免完整加载 pipeline 占用过多内存
        # 如果内存允许，或者需要 pipeline 的其他功能，可以保留
        # 为了减少初始化时的内存占用，我们分别加载tokenizer和processor
        self.processor = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_path,
            use_safetensors=True,
            torch_dtype=torch.float16, # 指定torch_dtype以减少内存
            low_cpu_mem_usage=True # 尝试进一步减少CPU内存使用
        ).image_processor

        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer" # SD3 medium的tokenizer通常在 'tokenizer', 'tokenizer_2', 'tokenizer_3'
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )
        self.tokenizer_three = T5Tokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_3"
        )
        
        # 显式卸载临时的完整pipeline（如果之前加载了）
        # import gc
        # del sd3_pipeline_temp 
        # gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = img.size
        crop = min(w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        pixel_values = self.processor.preprocess(img)[0] # processor返回的是list
        prompt = self.captions[idx]

        input_ids_one = self.tokenizer_one(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_one.model_max_length,
        ).input_ids[0]

        input_ids_two = self.tokenizer_two(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_two.model_max_length,
        ).input_ids[0]

        input_ids_three = self.tokenizer_three(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_three.model_max_length,
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "input_ids_three": input_ids_three,
        }


def main(train_configs):
    if train_configs.output_dir is not None:
        os.makedirs(os.path.abspath(train_configs.output_dir), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(train_configs.output_dir, "logs"))
    global_step = 0

    # 加载模型组件时使用 float16 并移到 device
    # 注意：pipeline 本身可能不需要在训练循环中完整保留，我们主要需要其组件
    # 为了节省 VRAM，我们分别加载组件
    vae = StableDiffusion3Pipeline.from_pretrained(
        train_configs.pretrained_model_path, use_safetensors=True, torch_dtype=torch.float16
    ).vae.to(device)
    
    mmdit = StableDiffusion3Pipeline.from_pretrained(
        train_configs.pretrained_model_path, use_safetensors=True, torch_dtype=torch.float16
    ).transformer.to(device)

    text_encoder_one = CLIPTextModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_3",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    # 冻结原始模型参数
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    for param in mmdit.parameters():
        param.requires_grad = False

    # 在MMDiT的注意力模块上配置LoRA
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 确保这些是MMDiT中正确的注意力层名称
        init_lora_weights="gaussian",
        bias="none", # 或者 "lora_only" 如果你希望LoRA层有可训练偏置
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    ghibli_dataset = Text2ImageDataset(
        train_configs.image_dir,
        train_configs.captions_dir,
        train_configs.pretrained_model_path,
        train_configs.resolution,
    )
    dataloader = DataLoader(
        ghibli_dataset, batch_size=train_configs.train_batch_size, shuffle=True
    )
    
    # 优化器只包含LoRA参数
    optimizer = torch.optim.AdamW(
        list(mmdit.parameters()), 
        lr=train_configs.learning_rate
    )

    # 训练循环
    mmdit.train() # LoRA层需要设置为训练模式

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )

    print("\n开始训练循环...")
    for epoch in range(train_configs.epochs):
        epoch_loss = 0.0
        for step, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"
        ):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            input_ids_three = batch["input_ids_three"].to(device)

            vae.eval() 
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (
                    latents * vae.config.scaling_factor
                ).to(torch.float16)

            # 文本编码 (SD3 的多文本编码器处理)
            text_encoder_one.eval()
            text_encoder_two.eval()
            text_encoder_three.eval()
            with torch.no_grad():  # 文本编码器本身是冻结的
                # CLIP Text Encoder L
                text_encoder_one_output = text_encoder_one(
                    input_ids_one, return_dict=True
                )
                # Sequence-level output for CLIP L (B, L, 768)
                text_embeds_one = text_encoder_one_output.last_hidden_state 
                # Pooled output for CLIP L (B, 768) - commonly taken from EOS token for CLIPTextModel
                pooled_prompt_embeds_one = text_embeds_one[:, -1, :] 

                # CLIP Text Encoder G (带投影)
                text_encoder_two_output = text_encoder_two(
                    input_ids_two, return_dict=True
                )
                text_embeds_two = text_encoder_two_output.last_hidden_state # Shape: (B, L2, 1280)
                # pooled_prompt_embeds 来自 CLIP-G text_embeds 属性 (通常是 [EOS] token 的投影)
                pooled_prompt_embeds_two = text_encoder_two_output.text_embeds # Shape: (B, 1280)


                # **关键：拼接两个 CLIP 的 pooled outputs**
                pooled_projections_final = torch.cat(
                    [pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1
                ).to(dtype=torch.float16)
                # 此时 pooled_projections_final 形状应为 (B, 768 + 1280) = (B, 2048)

                # T5 Text Encoder XL (或 SD3 Medium 使用的精简版T5)
                text_encoder_three_output = text_encoder_three(
                    input_ids_three, return_dict=True
                )
                # Sequence-level output for T5 (B, L_T5, 4096)
                text_embeds_three = text_encoder_three_output.last_hidden_state

                # --- Construct `encoder_hidden_states` for MMDiT ---
                # 1. Concatenate sequence-level CLIP embeddings along feature dim
                clip_combined_sequence_embeds = torch.cat(
                    [text_embeds_one, text_embeds_two], dim=-1
                ).to(dtype=torch.float16)
                # Expected shape: (B, L_CLIP, 768 + 1280) = (B, 77, 2048)

                # 2. Pad clip_combined_sequence_embeds to match T5's feature dimension (4096)
                # Calculate padding amount (target_dim - current_dim)
                padding_amount = text_embeds_three.shape[-1] - clip_combined_sequence_embeds.shape[-1]
                if padding_amount < 0:
                    raise ValueError(f"CLIP combined sequence embeds ({clip_combined_sequence_embeds.shape[-1]}) "
                                     f"is larger than T5 sequence embeds ({text_embeds_three.shape[-1]}). "
                                     f"This should not happen based on SD3 architecture.")

                # Apply padding along the feature dimension (-1)
                clip_combined_sequence_embeds_padded = torch.nn.functional.pad(
                    clip_combined_sequence_embeds, (0, padding_amount), mode='constant', value=0
                )
                # Expected shape after padding: (B, 77, 4096)

                # 3. Concatenate the padded CLIP embeddings and T5 embeddings along the sequence length dimension (-2)
                encoder_hidden_states = torch.cat(
                    [clip_combined_sequence_embeds_padded, text_embeds_three], dim=-2
                ).to(dtype=torch.float16)
                # Expected shape: (B, L_CLIP + L_T5, 4096)
                


            # 生成噪声
            noise = torch.randn_like(latents).to(device)

            # 采样时间步
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            # Flow Matching 的 t_cond (时间条件参数)
            # shift 通常是1, num_train_timesteps 通常是1000
            t_cond = (
                timesteps.float() + noise_scheduler.config.shift 
            ) / noise_scheduler.config.num_train_timesteps
            t_cond = (
                t_cond.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).to(torch.float16) 

            # 构建 MMDiT 的输入 (noisy_latents)
            # x_t = (1-t) * x_0 + t * epsilon  (这里 x_0=latents, epsilon=noise)
            noisy_latents = (1 - t_cond) * latents + t_cond * noise

            # 计算损失目标 (target): MMDiT 预测的是从 latents 到 noise 的速度场。
            # 目标速度场是 `noise - latents` (即 epsilon - x_0)。
            target = (noise - latents).to(torch.float16)

            # 前向通过 MMDiT (Transformer)
            print("------------------------------------------------------------------")
            print("noisy_latents data type:{0},shape:{1}".format(noisy_latents.dtype,noisy_latents.shape))
            print("timesteps data type:{0},shape:{1}".format(timesteps.dtype,timesteps.shape))
            print("encoder_hidden_states data type:{0},shape:{1}".format(encoder_hidden_states.dtype,encoder_hidden_states.shape))
            print("pooled_projections_final data type:{0},shape:{1}".format(pooled_projections_final.dtype,pooled_projections_final.shape))
            model_pred = mmdit(
                hidden_states=noisy_latents,
                timestep=timesteps, # 注意：这里传入的是离散的 timesteps 索引
                encoder_hidden_states=encoder_hidden_states, # (B, L_CLIP + L_T5, 4096)
                pooled_projections=pooled_projections_final,     # (B, 2048)
                return_dict=False,
            )[0]

            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if step % 10 == 0: # 每10步打印一次日志
                print(
                    f"Epoch {epoch+1}/{train_configs.epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )
        
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")
        
        # 保存 LoRA 检查点
        # PEFT模型可以直接调用save_pretrained保存LoRA权重
        lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
        mmdit.save_pretrained(lora_save_path)
        print(f"LoRA 权重已保存至: {lora_save_path}")

    # 最终保存
    final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
    mmdit.save_pretrained(final_lora_save_path)
    writer.close()
    print("训练完成. LoRA 权重已保存至", final_lora_save_path)


if __name__ == "__main__":
    # 确保配置文件路径正确
    train_parameters_config_path = "./lora_sd3_config.json" 
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()
        
    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)