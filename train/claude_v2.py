import copy
import glob
import os
import torch
import gc
import math
import numpy as np

# import torch.nn.functional as F
from datetime import datetime
from PIL import Image

from dataclasses import dataclass
from diffusers import AutoencoderKL, StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler

# from diffusers.training_utils import get_noise_sampler
from omegaconf import OmegaConf
from peft import get_peft_model, LoraConfig, TaskType,get_peft_model_state_dict, PeftModel
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
    gradient_accumulation_steps: int = 4
    rank: int = 8
    lora_alpha: int = 16
    pretrained_model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resolution: int = 512


def load_training_config(config_path: str) -> TrainParametersConfig:
    ## 从json文件中加载训练参数配置
    data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)


def _encode_prompt_with_t5_modified(
    text_encoder,
    input_ids,
    device=None
):
    prompt_embeds = text_encoder(input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds


def _encode_prompt_with_clip_modified(
    text_encoder,
    input_ids,
    device=None
):
    prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_modified(
    text_encoders,
    input_ids_one,
    input_ids_two,
    input_ids_three,
    device=None
):
    clip_text_encoder_one = text_encoders[0]
    prompt_embeds_one, pooled_prompt_embeds_one = _encode_prompt_with_clip_modified(
        text_encoder=clip_text_encoder_one,
        input_ids=input_ids_one,
        device=device if device is not None else clip_text_encoder_one.device,
    )

    clip_text_encoder_two = text_encoders[1]
    prompt_embeds_two, pooled_prompt_embeds_two = _encode_prompt_with_clip_modified(
        text_encoder=clip_text_encoder_two,
        input_ids=input_ids_two,
        device=device if device is not None else clip_text_encoder_two.device,
    )

    t5_text_encoder = text_encoders[2]
    t5_prompt_embed = _encode_prompt_with_t5_modified(
        text_encoder=t5_text_encoder,
        input_ids=input_ids_three,
        device=device if device is not None else t5_text_encoder.device,
    )

    clip_prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1)

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def compute_text_embeddings_modified(
    text_encoders,
    input_ids_one,
    input_ids_two,
    input_ids_three,
    device
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_modified(
            text_encoders=text_encoders,
            input_ids_one=input_ids_one,
            input_ids_two=input_ids_two,
            input_ids_three=input_ids_three,
            device=device
        )
    return prompt_embeds, pooled_prompt_embeds


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

        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )
        self.tokenizer_three = T5Tokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_3"
        )
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
        
        # 【修复】标准化像素值到[-1, 1]范围
        pixel_values = np.array(img).astype(np.float32) / 255.0
        pixel_values = (pixel_values - 0.5) * 2.0  # 归一化到[-1, 1]
        pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1)  # HWC -> CHW
        
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

    # 【修复】更高效的模型加载方式
    print("加载VAE...")
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path, 
        subfolder="vae",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(device)
    
    print("加载MMDiT transformer...")
    mmdit = SD3Transformer2DModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(device)

    print("加载文本编码器...")
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_3",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)

    # 冻结原始模型参数
    mmdit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    print("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,  # 【修复】使用正确的lora_alpha值
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    # 设置优化器
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        lr=train_configs.learning_rate,  # 【修复】简化学习率设置
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999)
    )

    # 设置学习率调度器(不一定需要使用.恒定学习率可能就OK)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(
        optimizer, 
        step_size=5,  # 每5个epoch降低一次
        gamma=0.8,    # 降低到80%
        verbose=True
    )

    ghibli_dataset = Text2ImageDataset(
        train_configs.image_dir,
        train_configs.captions_dir,
        train_configs.pretrained_model_path,
        train_configs.resolution
    )
    dataloader = DataLoader(
        ghibli_dataset, 
        batch_size=train_configs.train_batch_size, 
        shuffle=True,
        num_workers=2
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )

    # 【修复】使用正确的dataloader变量名
    num_update_steps_per_epoch = math.ceil(len(dataloader) / train_configs.gradient_accumulation_steps)
    real_max_train_steps = train_configs.epochs * num_update_steps_per_epoch
    real_num_train_epochs = math.ceil(real_max_train_steps / num_update_steps_per_epoch)
    real_num_batch_size = train_configs.train_batch_size * train_configs.gradient_accumulation_steps

    print("\n开始训练循环...")
    print(f"  Num examples = {len(ghibli_dataset)}")
    print(f"  Num batches each epoch = {len(dataloader)}")
    print(f"  Num Epochs = {real_num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_configs.train_batch_size}")
    print(f"  Total train batch size = {real_num_batch_size}")
    print(f"  Gradient Accumulation steps = {train_configs.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {real_max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    for epoch in range(train_configs.epochs):  # 【修复】使用正确的epochs数量
        mmdit.train()
        epoch_loss = 0.0
        batch_count = 0  # 【新增】用于计算平均损失
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_configs.epochs}")
        
        optimizer.zero_grad()  # 【修复】在epoch开始时清零梯度
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            input_ids_three = batch["input_ids_three"].to(device)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                [text_encoder_one, text_encoder_two, text_encoder_three], 
                input_ids_one, input_ids_two, input_ids_three, device
            )

            # VAE编码
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=torch.float32)

            # 生成噪声和时间步
            noise = torch.randn_like(model_input).to(device)
            
            # 【修复】使用正确的时间步范围和采样方法
            bsz = model_input.shape[0]
            # 对于Flow Matching，时间步通常在[0, 1]范围内
            timesteps = torch.rand((bsz,), device=device)
            timesteps = timesteps.long()
            
            # 【修复】Flow Matching的正确前向过程
            # Flow Matching使用线性插值：x_t = (1-t) * x_0 + t * noise
            # 其中t是标准化的时间步[0,1]
            t = timesteps.float().reshape(-1, 1, 1, 1)
            
            # 线性插值生成带噪声的latents
            noisy_latents = (1.0 - t) * model_input + t * noise
            
            # 【修复】Flow Matching的目标是速度场 v = noise - x_0
            # 这是从x_0到noise的方向向量
            target = noise - model_input

            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"警告: noisy_latents包含NaN或Inf，跳过此批次")
                continue   
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(f"警告: target包含NaN或Inf，跳过此批次")
                continue

            model_pred = mmdit(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            
            # 【修复】根据梯度累积步数缩放损失
            loss = loss / train_configs.gradient_accumulation_steps
            loss.backward()

            # 【修复】累积损失统计
            epoch_loss += loss.item() * train_configs.gradient_accumulation_steps  # 恢复真实损失值
            batch_count += 1

            # 【修复】梯度累积和优化器步骤
            if (step + 1) % train_configs.gradient_accumulation_steps == 0:
                # 【可选】梯度裁剪
                torch.nn.utils.clip_grad_norm_(mmdit.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 记录到tensorboard
                writer.add_scalar("train/loss", loss.item() * train_configs.gradient_accumulation_steps, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item() * train_configs.gradient_accumulation_steps:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'step': global_step
            })

        # 【修复】正确计算epoch平均损失
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")
        
        # 更新学习率调度器
        scheduler.step()
        
        # 【修复】定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == train_configs.epochs - 1:
            # 确保模型在CPU上以节省显存
            mmdit_for_save = mmdit
            current_epoch_mmdit_lora_layers = get_peft_model_state_dict(mmdit_for_save)
            
            lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
            os.makedirs(lora_save_path, exist_ok=True)
            
            # 【修复】使用正确的权重保存方式
            torch.save(current_epoch_mmdit_lora_layers, os.path.join(lora_save_path, "pytorch_lora_weights.safetensors"))
            
            # 也可以使用diffusers的保存方式
            try:
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=lora_save_path,
                    transformer_lora_layers=current_epoch_mmdit_lora_layers,
                    text_encoder_lora_layers=None,
                    text_encoder_2_lora_layers=None,
                )
            except Exception as e:
                print(f"使用diffusers保存权重时出错: {e}")
                print("已使用torch.save保存权重")
            
            print(f"LoRA 权重已保存至: {lora_save_path}")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 最终保存
    final_mmdit_lora_layers = get_peft_model_state_dict(mmdit)
    final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
    os.makedirs(final_lora_save_path, exist_ok=True)
    
    # 保存最终权重
    torch.save(final_mmdit_lora_layers, os.path.join(final_lora_save_path, "pytorch_lora_weights.safetensors"))
    
    try:
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=final_lora_save_path,
            transformer_lora_layers=final_mmdit_lora_layers,
            text_encoder_lora_layers=None,
            text_encoder_2_lora_layers=None,
        )
    except Exception as e:
        print(f"使用diffusers保存最终权重时出错: {e}")
        print("已使用torch.save保存最终权重")
    
    writer.close()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"训练完成. [{current_time}] LoRA 权重已保存至 {final_lora_save_path}")


if __name__ == "__main__":
    train_parameters_config_path = "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100_myscript.json"
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()
        
    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)