import copy
import glob
import os
import torch
import gc

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
    # 【关键修复】创建tokenizer引用供dataset使用  
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
        
        # 【修复】标准化像素值到[-1, 1]范围
        import numpy as np
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
    global_step = 0

    # 【修复】更高效的模型加载方式
    print("加载VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path, 
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    print("加载MMDiT transformer...")
    from diffusers import SD3Transformer2DModel
    mmdit = SD3Transformer2DModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    print("加载文本编码器...")
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
    mmdit.requires_grad_(False)

    # 【修复】确保LoRA目标模块名称正确
    print("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # SD3 MMDiT的标准注意力层
        init_lora_weights="gaussian",
        bias="none",
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
        ghibli_dataset, 
        batch_size=train_configs.train_batch_size, 
        shuffle=True,
        num_workers=2,  # 【修复】添加多进程数据加载
        pin_memory=True
    )
    
    # 优化器只包含LoRA参数
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),  # 【修复】只获取可训练参数
        lr=train_configs.learning_rate,
        weight_decay=0.01  # 【修复】添加权重衰减
    )

    # 【修复】添加学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_configs.epochs * len(dataloader),
        eta_min=train_configs.learning_rate * 0.1
    )

    # 训练循环
    mmdit.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )

    print("\n开始训练循环...")
    for epoch in range(train_configs.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_configs.epochs}")
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            input_ids_three = batch["input_ids_three"].to(device)

            # VAE编码
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 文本编码
            with torch.no_grad():
                # CLIP Text Encoder L
                text_encoder_one_output = text_encoder_one(
                    input_ids_one, return_dict=True
                )
                text_embeds_one = text_encoder_one_output.last_hidden_state 
                
                # 【关键修复】CLIPTextModel没有pooler_output，需要手动计算
                # 使用最后一个非padding token的hidden state作为pooled embedding
                # 对于CLIP，通常使用序列中最后一个有效token
                attention_mask_one = (input_ids_one != 49407).float()  # 49407是CLIP的pad_token_id
                sequence_lengths = attention_mask_one.sum(dim=1).long() - 1
                batch_indices = torch.arange(text_embeds_one.shape[0], device=text_embeds_one.device)
                pooled_prompt_embeds_one = text_embeds_one[batch_indices, sequence_lengths]

                # CLIP Text Encoder G
                text_encoder_two_output = text_encoder_two(
                    input_ids_two, return_dict=True
                )
                text_embeds_two = text_encoder_two_output.last_hidden_state
                pooled_prompt_embeds_two = text_encoder_two_output.text_embeds

                # 拼接pooled outputs
                pooled_projections_final = torch.cat(
                    [pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1
                ).to(dtype=torch.float16)

                # T5 Text Encoder
                text_encoder_three_output = text_encoder_three(
                    input_ids_three, return_dict=True
                )
                text_embeds_three = text_encoder_three_output.last_hidden_state

                # 构建encoder_hidden_states
                clip_combined_sequence_embeds = torch.cat(
                    [text_embeds_one, text_embeds_two], dim=-1
                ).to(dtype=torch.float16)

                # 填充到T5维度
                padding_amount = text_embeds_three.shape[-1] - clip_combined_sequence_embeds.shape[-1]
                if padding_amount > 0:
                    clip_combined_sequence_embeds_padded = torch.nn.functional.pad(
                        clip_combined_sequence_embeds, (0, padding_amount), mode='constant', value=0
                    )
                else:
                    clip_combined_sequence_embeds_padded = clip_combined_sequence_embeds

                encoder_hidden_states = torch.cat(
                    [clip_combined_sequence_embeds_padded, text_embeds_three], dim=-2
                ).to(dtype=torch.float16)

            # 生成噪声和时间步
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            # 【关键修复】SD3 Flow Matching的正确实现
            # 使用噪声调度器的正确方法
            sigmas = noise_scheduler.sigmas[timesteps.cpu()].to(device)
            sigmas = sigmas.reshape(-1, 1, 1, 1)
            
            # SD3使用的是velocity parameterization
            # v = alpha_t * noise - sigma_t * latents
            # 其中 alpha_t 和 sigma_t 是从调度器获取的
            alphas = (1.0 - sigmas).to(torch.float16)
            
            # 构建噪声图像: x_t = alpha_t * x_0 + sigma_t * noise  
            noisy_latents = (alphas * latents + sigmas * noise).to(torch.float16)
            
            # SD3的目标是velocity: v = alpha_t * noise - sigma_t * latents
            target = (alphas * noise - sigmas * latents).to(torch.float16)


            # 打印数据类型
            # print("------------------------------------------------------------------")
            # print("noisy_latents data type:{0},shape:{1}".format(noisy_latents.dtype,noisy_latents.shape))
            # print("timesteps data type:{0},shape:{1}".format(timesteps.dtype,timesteps.shape))
            # print("encoder_hidden_states data type:{0},shape:{1}".format(encoder_hidden_states.dtype,encoder_hidden_states.shape))
            # print("pooled_projections_final data type:{0},shape:{1}".format(pooled_projections_final.dtype,pooled_projections_final.shape))
            model_pred = mmdit(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections_final,
                return_dict=False,
            )[0]

            # 计算损失
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )
            
            # 反向传播
            loss.backward()
            
            # 【修复】添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(mmdit.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # 【修复】更新学习率
            optimizer.zero_grad()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # 【修复】内存清理
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")
        
        # 【修复】定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == train_configs.epochs - 1:
            lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
            mmdit.save_pretrained(lora_save_path)
            print(f"LoRA 权重已保存至: {lora_save_path}")

    # 最终保存
    final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
    mmdit.save_pretrained(final_lora_save_path)
    writer.close()
    print("训练完成. LoRA 权重已保存至", final_lora_save_path)


if __name__ == "__main__":
    train_parameters_config_path = "./lora_sd3_config.json" 
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()
        
    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)