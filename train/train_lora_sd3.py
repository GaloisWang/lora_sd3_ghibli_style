import copy
import glob
import os
import torch
import gc
import math
import numpy as np

from datetime import datetime
from PIL import Image
from dataclasses import dataclass
from diffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.torch_utils import is_compiled_module


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

    mixed_precision: str = "fp16"  # 可以是 "no", "fp16", "bf16"
    seed: int = 1337
    dataloader_num_workers: int = 2
    max_grad_norm: float = 1.0
    save_every_n_epochs: int = 10


def load_training_config(config_path: str) -> TrainParametersConfig:
    """从json文件中加载训练参数配置"""
    data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)


def _encode_prompt_with_t5(text_encoder, input_ids, device=None):
    prompt_embeds = text_encoder(input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, input_ids, device=None):
    prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders, input_ids_one, input_ids_two, input_ids_three, device=None
):
    clip_text_encoder_one = text_encoders[0]
    prompt_embeds_one, pooled_prompt_embeds_one = _encode_prompt_with_clip(
        text_encoder=clip_text_encoder_one,
        input_ids=input_ids_one,
        device=device if device is not None else clip_text_encoder_one.device,
    )

    clip_text_encoder_two = text_encoders[1]
    prompt_embeds_two, pooled_prompt_embeds_two = _encode_prompt_with_clip(
        text_encoder=clip_text_encoder_two,
        input_ids=input_ids_two,
        device=device if device is not None else clip_text_encoder_two.device,
    )

    t5_text_encoder = text_encoders[2]
    t5_prompt_embed = _encode_prompt_with_t5(
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


def compute_text_embeddings(
    text_encoders, input_ids_one, input_ids_two, input_ids_three, device
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
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

        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )
        self.tokenizer_three = T5Tokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_3"
        )

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

    project_config = ProjectConfiguration(
        project_dir=train_configs.output_dir,
        logging_dir=os.path.join(train_configs.output_dir, "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=train_configs.gradient_accumulation_steps,
        mixed_precision=train_configs.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if train_configs.seed is not None:
        set_seed(train_configs.seed)

    logger = get_logger(__name__)

    if accelerator.is_main_process:
        if train_configs.output_dir is not None:
            os.makedirs(train_configs.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("加载VAE...")
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    vae = vae.to(accelerator.device)

    logger.info("加载MMDiT transformer...")
    mmdit = SD3Transformer2DModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )

    logger.info("加载文本编码器...")
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    text_encoder_three = T5EncoderModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_3",
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )

    text_encoder_one = text_encoder_one.to(accelerator.device)
    text_encoder_two = text_encoder_two.to(accelerator.device)
    text_encoder_three = text_encoder_three.to(accelerator.device)

    mmdit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()

    logger.info("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    mmdit = get_peft_model(mmdit, lora_config)

    if accelerator.is_main_process:
        mmdit.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        lr=train_configs.learning_rate,
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999),
    )

    dataset = Text2ImageDataset(
        train_configs.image_dir,
        train_configs.captions_dir,
        train_configs.pretrained_model_path,
        train_configs.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_configs.train_batch_size,
        shuffle=True,
        num_workers=train_configs.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / train_configs.gradient_accumulation_steps
    )
    max_train_steps = train_configs.epochs * num_update_steps_per_epoch

    mmdit, optimizer, dataloader = accelerator.prepare(mmdit, optimizer, dataloader)

    if accelerator.is_main_process:
        accelerator.init_trackers("sd3_lora_training")

    logger.info("***** 训练信息 *****")
    logger.info(f"  数据集样本数 = {len(dataset)}")
    logger.info(f"  训练轮数 = {train_configs.epochs}")
    logger.info(f"  单设备批次大小 = {train_configs.train_batch_size}")
    logger.info(
        f"  总批次大小 = {train_configs.train_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  梯度累积步数 = {train_configs.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {max_train_steps}")
    logger.info(f"  混合精度 = {train_configs.mixed_precision}")

    global_step = 0
    first_epoch = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, train_configs.epochs):
        mmdit.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{train_configs.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(mmdit):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                input_ids_one = batch["input_ids_one"]
                input_ids_two = batch["input_ids_two"]
                input_ids_three = batch["input_ids_three"]
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    text_encoders=[
                        text_encoder_one,
                        text_encoder_two,
                        text_encoder_three,
                    ],
                    input_ids_one=input_ids_one,
                    input_ids_two=input_ids_two,
                    input_ids_three=input_ids_three,
                    device=accelerator.device,
                )

                with torch.no_grad():
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )

                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_latents = (1.0 - sigmas) * model_input + sigmas * noise

                model_pred = mmdit(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="logit_normal", sigmas=sigmas
                )
                model_pred = model_pred * (-sigmas) + noisy_latents
                target = model_input

                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if torch.isnan(loss):
                    logger.warning("损失为NaN,跳过此批次")
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        mmdit.parameters(), train_configs.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad()

            logs = {"train_loss": loss.detach().item()}
            if accelerator.sync_gradients:
                global_step += 1
                logs["step"] = global_step

            epoch_loss += loss.detach().item()
            accelerator.log(logs, step=global_step)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.4f}",
                    "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                    "step": global_step,
                }
            )

        avg_loss = epoch_loss / len(dataloader)
        accelerator.log({"train_epoch_loss": avg_loss}, step=epoch)
        logger.info(f"Epoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}")

        if accelerator.is_main_process and (
            (epoch + 1) % train_configs.save_every_n_epochs == 0
            or epoch == train_configs.epochs - 1
        ):
            unwrapped_mmdit = accelerator.unwrap_model(mmdit)
            lora_state_dict = get_peft_model_state_dict(unwrapped_mmdit)
            cleaned_state_dict = {}
            for key, value in lora_state_dict.items():
                if key.startswith("base_model.model."):
                    new_key = key.replace("base_model.model.", "")
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value

            lora_save_path = os.path.join(
                train_configs.output_dir, f"lora_epoch_{epoch+1}"
            )
            os.makedirs(lora_save_path, exist_ok=True)

            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=cleaned_state_dict,
                text_encoder_lora_layers=None,
                text_encoder_2_lora_layers=None,
            )
            logger.info(f"LoRA 权重已保存至: {lora_save_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_mmdit = accelerator.unwrap_model(mmdit)
        final_lora_state_dict = get_peft_model_state_dict(unwrapped_mmdit)
        cleaned_state_dict = {}
        for key, value in final_lora_state_dict.items():
            if key.startswith("base_model.model."):
                new_key = key.replace("base_model.model.", "")
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
        os.makedirs(final_lora_save_path, exist_ok=True)

        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=final_lora_save_path,
            transformer_lora_layers=cleaned_state_dict,
            text_encoder_lora_layers=None,
            text_encoder_2_lora_layers=None,
        )

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"训练完成. [{current_time}] LoRA 权重已保存至: {final_lora_save_path}"
        )

    accelerator.end_training()


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_parameters_config_path = (
        "/root/Codes/lora_diffusion/train/lora_sd3_config.json"
    )

    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()

    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)
