import copy
import glob
import os
import torch
import gc
import math

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
    rank: int = 8
    lora_alpha: int = 16
    pretrained_model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resolution: int = 512


def load_training_config(config_path: str) -> TrainParametersConfig:
    ## 从json文件中加载训练参数配置
    data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)

def print_cuda_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"[CUDA] Allocated: {allocated:.2f} MiB | Reserved: {reserved:.2f} MiB")

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
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path, 
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    print("加载MMDiT transformer...")
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
    
    # 优化器只包含LoRA参数 - 【关键修复1】进一步降低学习率
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),  # 【修复】只获取可训练参数
        lr=train_configs.learning_rate * 0.01,  # 【关键修复】进一步降低学习率到1%
        weight_decay=0.001,  # 【修复】减少权重衰减
        eps=1e-8,  # 【关键修复】增加epsilon防止除零
        betas=(0.9, 0.95)  # 【修复】降低beta2，减少momentum
    )

    # 【修复】添加学习率调度器 - 【关键修复2】使用更保守的调度器
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(
        optimizer, 
        step_size=5,  # 每5个epoch降低一次
        gamma=0.8,    # 降低到80%
        verbose=True
    )



    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )

    print("\n开始训练循环...")
    
    # 【关键修复3】减少梯度累积步数，增加更新频率
    gradient_accumulation_steps = 2  # 从4减少到2
    effective_batch_size = train_configs.train_batch_size * gradient_accumulation_steps
    
    for epoch in range(train_configs.epochs):
        # 训练循环
        mmdit.train()
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
            
            # 【关键修复4】更保守的时间步采样策略
            # 专注于中等时间步，避免极端情况
            min_timestep = int(0.1 * noise_scheduler.config.num_train_timesteps)  # 从10%开始
            max_timestep = int(0.9 * noise_scheduler.config.num_train_timesteps)  # 到90%结束
            timesteps = torch.randint(
                min_timestep,
                max_timestep,
                (latents.shape[0],),
                device=device,
            ).long()

            # 【关键修复5】SD3 Flow Matching的正确实现 - 添加数值稳定性检查
            try:
                # 使用噪声调度器的正确方法
                sigmas = noise_scheduler.sigmas[timesteps.cpu()].to(device)
                sigmas = sigmas.reshape(-1, 1, 1, 1)
                
                # 【关键修复】添加数值稳定性检查
                sigmas = torch.clamp(sigmas, min=1e-6, max=1.0 - 1e-6)
                
                # SD3使用的是velocity parameterization
                # v = alpha_t * noise - sigma_t * latents
                # 其中 alpha_t 和 sigma_t 是从调度器获取的
                alphas = (1.0 - sigmas).to(torch.float16)
                alphas = torch.clamp(alphas, min=1e-6, max=1.0 - 1e-6)
                
                # 构建噪声图像: x_t = alpha_t * x_0 + sigma_t * noise  
                noisy_latents = (alphas * latents + sigmas * noise).to(torch.float16)
                
                # SD3的目标是velocity: v = alpha_t * noise - sigma_t * latents
                target = (alphas * noise - sigmas * latents).to(torch.float16)

                # 【关键修复6】检查输入是否包含NaN或Inf
                if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                    print(f"警告: noisy_latents包含NaN或Inf，跳过此批次")
                    continue
                    
                if torch.isnan(target).any() or torch.isinf(target).any():
                    print(f"警告: target包含NaN或Inf，跳过此批次")
                    continue

                model_pred = mmdit(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections_final,
                    return_dict=False,
                )[0]

                # 【关键修复7】检查模型输出
                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                    print(f"警告: model_pred包含NaN或Inf，跳过此批次")
                    continue

                # 计算损失 - 【关键修复8】使用Huber损失增加稳定性
                loss = torch.nn.functional.huber_loss(
                    model_pred.float(), target.float(), delta=1.0, reduction="mean"
                )
                
                # 【关键修复9】检查损失值
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 损失值为NaN或Inf，跳过此批次")
                    continue
                    
                # 【关键修复10】限制损失值范围
                if loss.item() > 10.0:
                    print(f"警告: 损失值过大 ({loss.item():.4f})，跳过此批次")
                    continue
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 每gradient_accumulation_steps步或最后一步更新参数
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    # 【修复】添加梯度裁剪 - 【关键修复11】更严格的梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(mmdit.parameters(), max_norm=0.1)  # 进一步降低
                    
                    # 检查梯度范数
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print(f"警告: 梯度范数为NaN或Inf，跳过参数更新")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation_steps
                writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'grad_norm': f'{grad_norm:.4f}'
                })

            except Exception as e:
                print(f"训练步骤出错: {e}")
                optimizer.zero_grad()
                continue

            # 【修复】内存清理
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")
        
        # 【关键修复12】更新学习率调度器
        scheduler.step()  # StepLR每个epoch都调用
        
        # 【关键修复13】早停机制
        if avg_loss > 2.0:  # 如果损失过大，提前停止
            print(f"损失过大 ({avg_loss:.4f})，提前停止训练")
            break
        
        # 【修复】定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == train_configs.epochs - 1:
            current_epoch_mmdit = mmdit.to(torch.float32)
            current_epoch_mmdit_lora_layers = get_peft_model_state_dict(current_epoch_mmdit)
            lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
            os.makedirs(lora_save_path, exist_ok=True)
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=current_epoch_mmdit_lora_layers,
                text_encoder_lora_layers=None,
                text_encoder_2_lora_layers=None,
            )
            print(f"LoRA 权重已保存至: {lora_save_path}")


        print("当前{epoch}运行完毕后,CUDA显存情况是:")
        print_cuda_usage()            

    # 最终保存
    final_mmdit = mmdit.to(torch.float32)
    final_mmdit_lora_layers = get_peft_model_state_dict(final_mmdit)
    final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
    os.makedirs(final_lora_save_path, exist_ok=True)
    StableDiffusion3Pipeline.save_lora_weights(
        save_directory=final_lora_save_path,
        transformer_lora_layers=final_mmdit_lora_layers,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
    )
    writer.close()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"训练完成. [{current_time}] LoRA 权重已保存至", final_lora_save_path)


if __name__ == "__main__":
    train_parameters_config_path = "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100_myscript.json"
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()
        
    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)