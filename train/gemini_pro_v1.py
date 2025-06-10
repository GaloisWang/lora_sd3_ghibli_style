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



def main(train_configs):
    if train_configs.output_dir is not None:
        os.makedirs(os.path.abspath(train_configs.output_dir), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(train_configs.output_dir, "logs"))

    print("加载VAE...")
    vae = AutoencoderKL.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.float32, # 训练时通常用float32
        use_safetensors=True
    ).to(device)

    print("加载MMDiT transformer...")
    mmdit = SD3Transformer2DModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="transformer",
        torch_dtype=torch.float32, # 训练时通常用float32
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

    mmdit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    print("配置LoRA...")
    lora_config = LoraConfig(
        r=train_configs.rank,
        lora_alpha=train_configs.lora_alpha, # 【修正】使用配置中的lora_alpha
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        #【注】这里的LR缩放比较激进，通常只乘以gradient_accumulation_steps或不乘，需根据实验调整
        lr=train_configs.learning_rate, # 通常在此处设置基础学习率
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999)
    )

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(
        optimizer,
        step_size=5, # 每5个epoch降低一次
        gamma=0.8,
        verbose=True
    )

    ghibli_dataset = Text2ImageDataset(
        train_configs.image_dir,
        train_configs.captions_dir,
        train_configs.pretrained_model_path,
        train_configs.resolution
    )
    dataloader = DataLoader( # 【修复】变量名统一
        ghibli_dataset,
        batch_size=train_configs.train_batch_size,
        shuffle=True,
        num_workers=2 # 可以根据你的CPU核心数调整
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_configs.pretrained_model_path, subfolder="scheduler"
    )

    # 【修复】变量名统一为 dataloader
    num_update_steps_per_epoch = math.ceil(len(dataloader) / train_configs.gradient_accumulation_steps)
    real_max_train_steps = train_configs.epochs * num_update_steps_per_epoch
    real_num_train_epochs = math.ceil(real_max_train_steps / num_update_steps_per_epoch)
    # real_num_batch_size = train_configs.train_batch_size * train_configs.gradient_accumulation_steps # 这是有效的总批次大小

    print("\n开始训练循环...")
    print(f"  Num examples = {len(ghibli_dataset)}")
    print(f"  Num batches each epoch = {len(dataloader)}") # 【修复】变量名统一
    print(f"  Num Epochs = {real_num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_configs.train_batch_size}")
    print(f"  Total train batch size (effective) = {train_configs.train_batch_size * train_configs.gradient_accumulation_steps}") # 修正打印信息
    print(f"  Gradient Accumulation steps = {train_configs.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {real_max_train_steps}")
    global_step = 0
    # first_epoch = 0 # 未使用
    # initial_global_step = 0 # 未使用

    # 清零一次梯度，以防万一
    optimizer.zero_grad()

    for epoch in range(real_num_train_epochs):
        mmdit.train()
        epoch_loss_sum = 0.0 # 【修复】用于累加一个epoch内的所有batch loss (未缩放的)
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{real_num_train_epochs}")

        for step, batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            input_ids_three = batch["input_ids_three"].to(device)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                [text_encoder_one, text_encoder_two, text_encoder_three], input_ids_one, input_ids_two, input_ids_three, device
            )

            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=torch.float32) # 确保在设备上且类型正确

            noise = torch.randn_like(model_input).to(device) # 确保noise在同一设备
            min_timestep = int(0.1 * noise_scheduler.config.num_train_timesteps)
            max_timestep = int(0.9 * noise_scheduler.config.num_train_timesteps)
            timesteps = torch.randint(
                min_timestep,
                max_timestep,
                (model_input.shape[0],),
                device=device, # 确保timesteps在同一设备
            ).long()

            sigmas = noise_scheduler.sigmas[timesteps.cpu()].to(device) # sigmas 需要先在cpu上索引，再移到device
            sigmas = sigmas.reshape(-1, 1, 1, 1)
            sigmas = torch.clamp(sigmas, min=1e-6) # 通常不需要max clamp, 除非sigmas本身可能 > 1

            # FlowMatch-specific target calculation (from SD3 paper or reference implementation)
            # noisy_latents = model_input + sigmas * noise # This is typical for some diffusion models
            # target = noise # if predicting noise
            # Or if predicting (x - noise)/sigma, etc. Let's stick to your original logic for FlowMatch target if it's correct for it.
            # Your original target logic:
            noisy_latents = (torch.sqrt(1.0 - sigmas**2) * model_input + sigmas * noise).to(torch.float32)
            target = (model_input - torch.sqrt(1.0 - sigmas**2) * noisy_latents) / sigmas # This is one way to define target for Flow Matching, ensure it matches the paper/scheduler expectation.
            # OR, more directly from your code:
            # alphas = (1.0 - sigmas).to(torch.float32) # This might not be the 'alpha' for FlowMatch, sigmas are directly used.
            # alphas = torch.clamp(alphas, min=1e-6, max=1.0 - 1e-6)
            # noisy_latents = (alphas * model_input + sigmas * noise).to(torch.float32) # This is more like DDPM noise schedule
            # target = (noise - model_input).to(torch.float32) # This target is for predicting (noise - x_0)

            # Let's assume your original sigmas, noisy_latents and target logic was what you intended for FlowMatch
            # Reverting to your original noisy_latents and target for now, but double check FlowMatch paper:
            sigmas_for_noise = sigmas # using sigmas as they are
            sigmas_for_signal = torch.sqrt(1.0 - sigmas**2) # often denoted as alphas

            noisy_latents = (sigmas_for_signal * model_input + sigmas_for_noise * noise).to(torch.float32)
            # For FlowMatch, the model often predicts the vector field u_t(x_t) which is (x_1 - x_0) if linear interpolation path
            # Or if it's about predicting velocity (dx/dt), and target is (x1 - noisy_latents) / (1-t) scaled by sigma related term.
            # The original code's target: target = (noise - model_input).to(torch.float32)
            # This implies the model is trying to predict (noise - model_input).
            # Let's use a common FlowMatch target: velocity.
            # If path is x_t = t * x_1 + (1-t) * x_0 (where x_0 is noise, x_1 is data)
            # then velocity v_t = x_1 - x_0.
            # If model predicts v_t, loss is (pred - (model_input - noise))^2
            # Your scheduler FlowMatchEulerDiscreteScheduler might expect a specific target.
            # Let's stick to *your* target definition to minimize changes beyond the fix request.
            # Your sigmas here are more like noise levels.
            # alphas = (1.0 - sigmas).to(torch.float32)
            # alphas = torch.clamp(alphas, min=1e-6, max=1.0 - 1e-6)
            # noisy_latents = (alphas * model_input + sigmas * noise).to(torch.float32) # This was your noisy_latents
            # target = (noise - model_input).to(torch.float32) # This was your target

            # Re-evaluating your noisy_latents and target based on your code structure:
            # sigmas are from noise_scheduler.sigmas. For FlowMatch, these usually represent std of noise at time t.
            # A common formulation for noisy image is: x_t = sqrt(alpha_t^2) * x_0 + sqrt(1-alpha_t^2) * noise
            # where alpha_t is signal rate, sqrt(1-alpha_t^2) is noise rate (sigma_t)
            # Your `sigmas` are the noise rates. So signal rates are `sqrt(1-sigmas^2)`
            # noisy_latents = torch.sqrt(1.0 - sigmas**2) * model_input + sigmas * noise # Consistent with common diffusion
            # And if model predicts noise:
            # target = noise
            # If model predicts data x0:
            # target = model_input
            # If model predicts velocity (as in consistency models or some flow matching variants):
            # target = (model_input - sigmas * noise) / torch.sqrt(1.0 - sigmas**2) # if noise is N(0,I) and x_t = alpha_t x_0 + sigma_t noise, predicting x_0 from x_t.
            # Or target = (model_input - torch.sqrt(1.0 - sigmas**2) * noisy_latents) / sigmas # if predicting noise from x_t.

            # Given your original code:
            # alphas = (1.0 - sigmas) # This is not standard for alpha in signal/noise rate context.
            # Let's assume your sigmas are directly the 't' or related to 't' in a flow t*x1 + (1-t)*x0
            # And the model is trying to predict something related to the "flow" or "difference".
            # Your original code:
            #   alphas = (1.0 - sigmas)
            #   noisy_latents = (alphas * model_input + sigmas * noise)
            #   target = (noise - model_input)
            # This implies model_pred should approximate (noise - model_input).
            # Let's use this to ensure minimal changes to your core diffusion logic, focusing on accumulation.

            # Using your original noising and target:
            alphas_from_sigmas = (1.0 - sigmas).to(torch.float32) # Your 'alphas'
            alphas_from_sigmas = torch.clamp(alphas_from_sigmas, min=1e-6, max=1.0 - 1e-6)
            current_noisy_latents = (alphas_from_sigmas * model_input + sigmas * noise).to(torch.float32)
            current_target = (noise - model_input).to(torch.float32)


            if torch.isnan(current_noisy_latents).any() or torch.isinf(current_noisy_latents).any():
                print(f"警告: noisy_latents包含NaN或Inf，跳过此批次")
                continue
            if torch.isnan(current_target).any() or torch.isinf(current_target).any():
                print(f"警告: target包含NaN或Inf，跳过此批次")
                continue

            model_pred = mmdit(
                hidden_states=current_noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # Calculate loss for the current batch
            batch_loss = torch.mean(
                ((model_pred.float() - current_target.float()) ** 2).reshape(current_target.shape[0], -1),
                1,
            )
            batch_loss = batch_loss.mean()

            # 【修复】累加当前batch的真实loss (用于epoch平均loss计算)
            epoch_loss_sum += batch_loss.item()

            # 【修复】为梯度累积缩放loss
            scaled_loss = batch_loss / train_configs.gradient_accumulation_steps
            scaled_loss.backward()

            # 【修复】梯度累积逻辑
            if (step + 1) % train_configs.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # 可以选择在这里进行梯度裁剪 (optional)
                # torch.nn.utils.clip_grad_norm_(mmdit.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad() # 清零梯度，为下一次累积做准备

            writer.add_scalar("train/batch_loss", batch_loss.item(), global_step) # 记录每个batch的真实loss
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss.item():.4f}', # 显示当前batch的真实loss
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 【修复】计算并记录epoch的平均loss
        avg_epoch_loss = epoch_loss_sum / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均损失: {avg_epoch_loss:.4f}")

        # 【修复】学习率调度器在每个epoch结束时调用
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == real_num_train_epochs - 1: # real_num_train_epochs anstead of train_configs.epochs
            # 保存LoRA权重
            # 注意: StableDiffusion3Pipeline.save_lora_weights 可能需要 pipeline 实例或特定组件
            # diffusers >= 0.27.0 开始可以直接使用 PeftModel的 save_pretrained
            # 或者使用 get_peft_model_state_dict
            lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
            os.makedirs(lora_save_path, exist_ok=True)

            # 获取LoRA权重字典
            current_epoch_mmdit_lora_layers = get_peft_model_state_dict(mmdit)

            # StableDiffusion3Pipeline 类本身没有 save_lora_weights 这个静态方法或类方法
            # 它通常是 pipeline 实例的方法。
            # 如果只想保存 mmdit 的 LoRA 权重，可以这样做：
            # mmdit.save_pretrained(lora_save_path) # 这会保存完整的LoRA模型配置和权重

            # 或者，如果你想遵循 StableDiffusionPipeline 的 save_lora_weights 格式，
            # 你需要构造一个 pipeline 并使用它的方法，或者手动保存。
            # 为了简单起见，且由于你只训练了 mmdit，我们直接保存 mmdit 的 LoRA 权重。
            # torch.save(current_epoch_mmdit_lora_layers, os.path.join(lora_save_path, "pytorch_lora_weights.safetensors"))
            # print(f"MMDiT LoRA 权重已保存至: {os.path.join(lora_save_path, 'pytorch_lora_weights.safetensors')}")
            # ^ 上述方法只保存了 state_dict，加载时需要先创建模型结构。

            # 使用 diffusers 推荐的方式 (如果仅训练 transformer):
            # 需要一个 base_model (mmdit 未应用lora前的状态) 和 peft_model (mmdit 应用lora后的状态)
            # 但 get_peft_model_state_dict 是更底层的。
            # 对于 SD3, save_lora_weights 的参数是 transformer_lora_layers, text_encoder_lora_layers 等
            # 既然我们只有 transformer 的 LoRA 层，其他可以是 None
            
            # 【修复】保存LoRA权重的正确方式，确保mmdit在CPU上且为float32以避免兼容问题
            mmdit_to_save = mmdit.to(torch.float32).cpu()
            state_dict_to_save = get_peft_model_state_dict(mmdit_to_save)
            
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=state_dict_to_save, # 直接传递state_dict
                # 如果文本编码器也用了LoRA，则传入它们的 state_dict
                text_encoder_lora_layers=None,
                text_encoder_2_lora_layers=None,
                text_encoder_3_lora_layers=None, # SD3 有三个文本编码器
                # vae_lora_layers=None # 如果VAE也用了LoRA
            )
            print(f"LoRA 权重已保存至: {lora_save_path}")
            mmdit.to(device) # 移回训练设备


    # 最终保存
    final_lora_save_path = os.path.join(train_configs.output_dir, "lora_final")
    os.makedirs(final_lora_save_path, exist_ok=True)
    
    mmdit_to_save_final = mmdit.to(torch.float32).cpu()
    final_mmdit_lora_layers = get_peft_model_state_dict(mmdit_to_save_final)

    StableDiffusion3Pipeline.save_lora_weights(
        save_directory=final_lora_save_path,
        transformer_lora_layers=final_mmdit_lora_layers,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
        text_encoder_3_lora_layers=None,
    )
    print(f"最终LoRA 权重已保存至: {final_lora_save_path}")
    
    writer.close()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"训练完成. [{current_time}]")


if __name__ == "__main__":
    train_parameters_config_path = "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100_myscript.json"
    if not os.path.exists(train_parameters_config_path):
        print(f"错误：配置文件 {train_parameters_config_path} 不存在！请创建配置文件。")
        exit()

    data_dict = OmegaConf.load(train_parameters_config_path)
    train_configs = TrainParametersConfig(**data_dict)

    main(train_configs)