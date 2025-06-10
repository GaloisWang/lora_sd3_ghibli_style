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
from torch.optim.lr_scheduler import ConstantLR
from dataclasses import dataclass
from diffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

# from diffusers.training_utils import get_noise_sampler
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
    # 添加一个用于AMP的参数，默认为True
    use_amp: bool = True


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

def print_cuda_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"[CUDA] Allocated: {allocated:.2f} MiB | Reserved: {reserved:.2f} MiB")



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

    # 设置自动混合精度数据类型，默认为float16
    if torch.cuda.is_available():
        # 检查是否支持bfloat16（A100、3090等支持）
        if torch.cuda.get_device_capability()[0] >= 8:
            # amp_dtype = torch.bfloat16  # 更好的数值稳定性
            amp_dtype = torch.float16  # 更好的数值稳定性
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    print("加载VAE...")
    # VAE的解码阶段可能对精度敏感，在训练阶段通常不训练VAE，可以保留FP32
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="vae",
        # vae在推理时通常用FP32，在训练时如果冻结，可以保持FP32。
        # 如果要训练vae，这里才需要考虑AMP
        torch_dtype=torch.float32, # 保持FP32以获得更好的数值稳定性，因为VAE通常不参与LoRA训练
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

    print("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    # 设置优化器
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        lr=train_configs.learning_rate,
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999),
    )
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)

    # **新增：PyTorch AMP 梯度缩放器**
    # 如果使用AMP，需要一个梯度缩放器来防止FP16梯度下溢。
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
        num_workers=2,
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
    print(f"  Using AMP: {train_configs.use_amp} (dtype: {amp_dtype})") # 打印AMP状态
    global_step = 0

    optimizer.zero_grad()

    for epoch in range(real_num_train_epochs):
        mmdit.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{real_num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            # 使用torch.autocast上下文管理器
            # 只有在GPU上且train_configs.use_amp为True时才启用autocast
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=train_configs.use_amp):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32) # VAE输入保持FP32

                input_ids_one = batch["input_ids_one"].to(device)
                input_ids_two = batch["input_ids_two"].to(device)
                input_ids_three = batch["input_ids_three"].to(device)

                # 计算文本 embeddings 和 pooled embeddings
                # 文本编码器不在autocast中，但其输出会被autocast自动转换到低精度
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                    text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                    input_ids_one=input_ids_one,
                    input_ids_two=input_ids_two,
                    input_ids_three=input_ids_three,
                    device=device,
                )

                # VAE编码 (在autocast内部，VAEE的forward会尝试以低精度运行，但如果VAE本身是FP32，它会保持FP32)
                # 因为VAE是冻结的，并且通常希望其输出精确，所以我们将其输入保持FP32，
                # VAE内部的计算如果模型是FP32则继续FP32。
                # autocast 会自动将VAE的输出（latent_dist.sample()）转换为amp_dtype。
                with torch.no_grad():
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    # model_input 现在在autocast的控制下，dtype会是amp_dtype

                # 【修复6】使用正确的SD3 Flow Matching噪声生成
                noise = torch.randn_like(model_input).to(device)
                bsz = model_input.shape[0]
                timesteps = torch.rand((bsz,), device=device)
                sigmas = timesteps.float()
                timesteps_scaled = (timesteps * 1000).long()

                t = sigmas.reshape(-1, 1, 1, 1)
                # 这里的线性插值会自动使用当前autocast的dtype
                noisy_latents = ((1.0 - t) * model_input + t * noise)
                target = (noise - model_input) # target也会自动是amp_dtype

                model_pred = mmdit(
                    hidden_states=noisy_latents,
                    timestep=timesteps_scaled,
                    encoder_hidden_states=prompt_embeds, # 这里的prompt_embeds和pooled_projections会自动匹配autocast的dtype
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

            # 损失计算在autocast外部进行，或者在autocast内部进行，但将模型预测和目标转换为FP32。
            # 推荐将损失计算放在autocast外部，并转换为FP32以避免精度问题。
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )

            # 检查损失是否为NaN (可能在autocast内部计算时出现，但我们在此处检查)
            if torch.isnan(loss):
                print(f"警告: 损失为NaN，跳过此批次")
                # 确保在跳过批次时清除梯度
                if (step + 1) % train_configs.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    optimizer.zero_grad()
                continue

            loss = loss / train_configs.gradient_accumulation_steps

            # **新增：梯度缩放**
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
                    scaler.unscale_(optimizer) # 在unscale之前裁剪梯度 (如果需要)
                    torch.nn.utils.clip_grad_norm_(mmdit.parameters(), 1.0) # 例如，添加梯度裁剪
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(mmdit.parameters(), 1.0) # 例如，添加梯度裁剪
                    optimizer.step()
                optimizer.zero_grad() # 每次优化器更新后清零梯度

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
                    "loss": f"{loss.detach().item() * train_configs.gradient_accumulation_steps:.4f}",
                    "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                    "step": global_step,
                }
            )

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")

        if (epoch + 1) % 1 == 0 or epoch == train_configs.epochs - 1:
            # 保存LoRA权重时，通常建议将LoRA层转换回FP32，因为它们的参数数量较少，
            # 并且可以确保推理时更稳定。
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
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("当前{epoch}运行完毕后,CUDA显存情况是:")
        print_cuda_usage()

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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_parameters_config_path = (
        "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100_myscript.json"
    )
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()

    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)