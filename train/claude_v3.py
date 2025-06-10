import copy
import glob
import os
import torch
import gc
import math
import numpy as np
from datetime import datetime
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR
from dataclasses import dataclass
from diffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from omegaconf import OmegaConf
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    get_peft_model_state_dict,
    PeftModel,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)


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
    # 添加一个用于AMP的参数，默认为True
    use_amp: bool = True
    # 新增：内存优化参数
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = True
    checkpointing: bool = True


def load_training_config(config_path: str) -> TrainParametersConfig:
    ## 从json文件中加载训练参数配置
    data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)


def _encode_prompt_with_t5_modified(text_encoder, input_ids, device=None):
    prompt_embeds = text_encoder(input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds


def _encode_prompt_with_clip_modified(text_encoder, input_ids, device=None):
    prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_modified(
    text_encoders, input_ids_one, input_ids_two, input_ids_three, device=None
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
    pooled_prompt_embeds = torch.cat(
        [pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def compute_text_embeddings_modified(
    text_encoders, input_ids_one, input_ids_two, input_ids_three, device
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_modified(
            text_encoders=text_encoders,
            input_ids_one=input_ids_one,
            input_ids_two=input_ids_two,
            input_ids_three=input_ids_three,
            device=device,
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

        # 优化：只加载必要的tokenizer，避免加载完整pipeline
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )
        self.tokenizer_three = T5Tokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_3"
        )
        
        # 清理内存
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

        # 标准化像素值到[-1, 1]范围
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

    # 设置自动混合精度数据类型，默认为float16
    if torch.cuda.is_available():
        # 检查是否支持bfloat16（A100、3090等支持）
        if torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16  # 更好的数值稳定性
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    print("加载VAE...")
    # VAE的解码阶段可能对精度敏感，保持FP32
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,  # 保持FP32以获得更好的数值稳定性
        use_safetensors=True,
    ).to(device)

    print("加载MMDiT transformer...")
    mmdit = SD3Transformer2DModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=amp_dtype,
        use_safetensors=True,
    ).to(device)

    print("加载文本编码器...")
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=amp_dtype,
        use_safetensors=True,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=amp_dtype,
        use_safetensors=True,
    ).to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_3",
        torch_dtype=amp_dtype,
        use_safetensors=True,
    ).to(device)

    # 冻结原始模型参数
    mmdit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # 设置文本编码器为评估模式
    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()

    # CPU卸载优化 - 针对文本编码器
    if train_configs.enable_cpu_offload:
        # 将文本编码器移到CPU以释放GPU内存
        text_encoder_one = text_encoder_one.to("cpu")
        text_encoder_two = text_encoder_two.to("cpu")
        text_encoder_three = text_encoder_three.to("cpu")
        print("文本编码器已移至CPU以节省GPU内存")

    print("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    # 启用梯度检查点以节省内存
    if train_configs.checkpointing:
        mmdit.gradient_checkpointing_enable()
        print("已启用梯度检查点")

    # 设置优化器
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        lr=train_configs.learning_rate,
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999),
    )
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)

    # PyTorch AMP 梯度缩放器
    scaler = torch.amp.GradScaler(enabled=train_configs.use_amp)

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
        num_workers=0,  # 减少内存使用，设为0
        pin_memory=False,  # 减少内存压力
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / train_configs.gradient_accumulation_steps
    )
    real_max_train_steps = train_configs.epochs * num_update_steps_per_epoch
    real_num_train_epochs = train_configs.epochs
    real_num_batch_size = (
        train_configs.train_batch_size * train_configs.gradient_accumulation_steps
    )

    print("\n开始训练循环...")
    print(f"  Num examples = {len(ghibli_dataset)}")
    print(f"  Num batches each epoch = {len(dataloader)}")
    print(f"  Num Epochs = {real_num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_configs.train_batch_size}")
    print(f"  Total train batch size = {real_num_batch_size}")
    print(
        f"  Gradient Accumulation steps = {train_configs.gradient_accumulation_steps}"
    )
    print(f"  Total optimization steps = {real_max_train_steps}")
    print(f"  Using AMP: {train_configs.use_amp} (dtype: {amp_dtype})")
    global_step = 0

    optimizer.zero_grad()

    for epoch in range(real_num_train_epochs):
        mmdit.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{real_num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            # 使用torch.autocast上下文管理器
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", 
                               dtype=amp_dtype, enabled=train_configs.use_amp):
                
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)

                input_ids_one = batch["input_ids_one"]
                input_ids_two = batch["input_ids_two"]
                input_ids_three = batch["input_ids_three"]

                # 计算文本 embeddings（如果启用CPU卸载，需要临时移动到GPU）
                if train_configs.enable_cpu_offload:
                    # 临时移动到GPU进行计算
                    text_encoder_one_temp = text_encoder_one.to(device)
                    text_encoder_two_temp = text_encoder_two.to(device)
                    text_encoder_three_temp = text_encoder_three.to(device)
                    
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                        text_encoders=[text_encoder_one_temp, text_encoder_two_temp, text_encoder_three_temp],
                        input_ids_one=input_ids_one,
                        input_ids_two=input_ids_two,
                        input_ids_three=input_ids_three,
                        device=device,
                    )
                    
                    # 立即移回CPU
                    text_encoder_one_temp = text_encoder_one_temp.to("cpu")
                    text_encoder_two_temp = text_encoder_two_temp.to("cpu")
                    text_encoder_three_temp = text_encoder_three_temp.to("cpu")
                    del text_encoder_one_temp, text_encoder_two_temp, text_encoder_three_temp
                else:
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                        text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                        input_ids_one=input_ids_one,
                        input_ids_two=input_ids_two,
                        input_ids_three=input_ids_three,
                        device=device,
                    )

                # VAE编码
                with torch.no_grad():
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor

                # 使用正确的SD3 Flow Matching噪声生成
                noise = torch.randn_like(model_input).to(device)
                bsz = model_input.shape[0]
                timesteps = torch.rand((bsz,), device=device)
                sigmas = timesteps.float()
                timesteps_scaled = (timesteps * 1000).long()

                t = sigmas.reshape(-1, 1, 1, 1)
                noisy_latents = ((1.0 - t) * model_input + t * noise)
                target = (noise - model_input)

                # 检查输入是否包含NaN或Inf
                if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                    print(f"警告: noisy_latents包含NaN或Inf，跳过此批次")
                    continue
                    
                if torch.isnan(target).any() or torch.isinf(target).any():
                    print(f"警告: target包含NaN或Inf，跳过此批次")
                    continue

                # 模型预测
                model_pred = mmdit(
                    hidden_states=noisy_latents,
                    timestep=timesteps_scaled,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # 检查模型输出
                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                    print(f"警告: model_pred包含NaN或Inf，跳过此批次")
                    continue

            # 损失计算在autocast外部进行
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"警告: 损失为NaN，跳过此批次")
                if (step + 1) % train_configs.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    optimizer.zero_grad()
                continue

            loss = loss / train_configs.gradient_accumulation_steps

            # 梯度缩放
            if train_configs.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * train_configs.gradient_accumulation_steps

            # 梯度累积与优化器更新
            if (step + 1) % train_configs.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(dataloader):
                if train_configs.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mmdit.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(mmdit.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar(
                    "train/loss",
                    loss.item() * train_configs.gradient_accumulation_steps,
                    global_step,
                )
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
                global_step += 1

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * train_configs.gradient_accumulation_steps:.4f}",
                    "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                    "step": global_step,
                }
            )

            # 定期清理内存
            if step % 5 == 0:  # 增加清理频率
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")

        # Epoch结束后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if (epoch + 1) % 1 == 0 or epoch == train_configs.epochs - 1:
            # 保存时临时转换为FP32
            current_epoch_mmdit_lora_layers = get_peft_model_state_dict(
                mmdit.to(torch.float32)
            )
            lora_save_path = os.path.join(
                train_configs.output_dir, f"lora_epoch_{epoch+1}"
            )
            os.makedirs(lora_save_path, exist_ok=True)
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=current_epoch_mmdit_lora_layers,
                text_encoder_lora_layers=None,
                text_encoder_2_lora_layers=None,
            )
            print(f"LoRA 权重已保存至: {lora_save_path}")
            
            # 保存后立即清理内存
            del current_epoch_mmdit_lora_layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # 最终保存
    final_mmdit_lora_layers = get_peft_model_state_dict(mmdit.to(torch.float32))
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
    train_parameters_config_path = (
        "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100_myscript.json"
    )
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()

    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)