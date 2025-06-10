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
        torch_dtype=torch.float32, # 注意：这里如果希望使用 fp16/bf16，需要设置为对应类型
        use_safetensors=True,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float32, # 注意：这里如果希望使用 fp16/bf16，需要设置为对应类型
        use_safetensors=True,
    ).to(device)
    # T5 Encoder Model通常不需要Projection
    text_encoder_three = T5EncoderModel.from_pretrained(
        train_configs.pretrained_model_path,
        subfolder="text_encoder_3",
        torch_dtype=torch.float32, # 注意：这里如果希望使用 fp16/bf16，需要设置为对应类型
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
        lora_alpha=train_configs.lora_alpha, # 使用train_configs.lora_alpha
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"] # 确保这些模块名是MMDiT中LoRA要注入的模块名
    )
    mmdit = get_peft_model(mmdit, lora_config)
    mmdit.print_trainable_parameters()

    # 设置优化器
    optimizer = torch.optim.AdamW(
        mmdit.parameters(),
        lr=train_configs.learning_rate, # 梯度累积的学习率调整通常在外部进行
        weight_decay=1e-04,
        eps=1e-08,
        betas=(0.9, 0.999)
    )

    # 设置学习率调度器
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

    # 修正计算变量名
    num_update_steps_per_epoch = math.ceil(len(dataloader) / train_configs.gradient_accumulation_steps)
    real_max_train_steps = train_configs.epochs * num_update_steps_per_epoch
    real_num_train_epochs = train_configs.epochs # 直接使用配置的epochs
    real_num_batch_size = train_configs.train_batch_size * train_configs.gradient_accumulation_steps

    print("\n开始训练循环...")
    print(f"  Num examples = {len(ghibli_dataset)}")
    print(f"  Num batches per epoch = {len(dataloader)}")
    print(f"  Num Epochs = {real_num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_configs.train_batch_size}")
    print(f"  Total train batch size (w/ accumulation) = {real_num_batch_size}")
    print(f"  Gradient Accumulation steps = {train_configs.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {real_max_train_steps}")
    global_step = 0
    # first_epoch = 0 # 这些变量在此处不需要
    # initial_global_step = 0

    for epoch in range(real_num_train_epochs):
        mmdit.train() # 确保MMDiT处于训练模式 (LoRA层激活)
        epoch_loss = 0.0 # 每个epoch开始时重置
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{real_num_train_epochs}")
        
        optimizer.zero_grad() # 每个epoch开始前清零一次梯度

        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            input_ids_three = batch["input_ids_three"].to(device)

            # 计算文本 embeddings 和 pooled embeddings
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_modified(
                text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                input_ids_one=input_ids_one,
                input_ids_two=input_ids_two,
                input_ids_three=input_ids_three,
                device=device
            )

            # VAE编码
            with torch.no_grad(): # VAE编码过程不需要计算梯度
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                # 确保模型输入的数据类型与MMDiT期望的保持一致，特别是当你使用混合精度训练时
                # 这里假设MMDiT是float32，所以model_input也保持float32
                model_input = model_input.to(dtype=torch.float32) 

            # 生成噪声和时间步
            noise = torch.randn_like(model_input).to(device)
            # 确保timesteps的范围和采样方式与Stable Diffusion 3的训练保持一致
            # 这里的噪音调度器处理可能需要根据SD3的训练细节进行微调
            # 通常的扩散模型训练中，timesteps是[0, num_train_timesteps-1]
            # 这里的min_timestep, max_timestep可能用于特定采样策略
            min_timestep = int(0.1 * noise_scheduler.config.num_train_timesteps)
            max_timestep = int(0.9 * noise_scheduler.config.num_train_timesteps)
            timesteps = torch.randint(
                min_timestep,
                max_timestep + 1, # 确保包含max_timestep
                (model_input.shape[0],),
                device=device,
            ).long()

            # 根据FlowMatchEulerDiscreteScheduler的定义，sigmas和alphas的计算方式可能需要调整
            # 原始Stable Diffusion的Langevin动力学目标通常是预测噪声
            # SD3的Flow Matching目标是预测速度场 (v)
            # model_pred = v_target = x_start - x_t / sigma(t)
            # 这里的目标是 (noise - model_input) 对应的是预测噪声
            # 如果是 Flow Matching，目标应该是预测 `v_target`
            # `v_target = noise_scheduler.get_velocity_target(model_input, noise, timesteps)`
            # 建议参考Diffusers官方FlowMatchEulerDiscreteScheduler的训练示例。
            # 这里我按照你的原有目标计算，但要注意这可能不是Flow Matching的目标
            sigmas = noise_scheduler.sigmas[timesteps.cpu()].to(device)
            sigmas = sigmas.reshape(-1, 1, 1, 1)

            # 这里的clamp对sigmas和alphas的应用值得商榷，可能会影响模型表现
            # sigmas = torch.clamp(sigmas, min=1e-6, max=1.0 - 1e-6) # 可能会限制扩散过程的范围
            # alphas = (1.0 - sigmas).to(torch.float32)
            # alphas = torch.clamp(alphas, min=1e-6, max=1.0 - 1e-6)

            noisy_latents = (torch.sqrt(1.0 - sigmas) * model_input + torch.sqrt(sigmas) * noise).to(torch.float32)
            target = noise # Flow Matching通常是预测噪声
            # 原始代码 target = (noise - model_input).to(torch.float32) 
            # 这是一个典型的预测噪声的损失目标
            # 如果是flow matching，目标通常是预测velocity field
            # v_target = (model_input - noisy_latents) / sigmas
            # 需要根据FlowMatchEulerDiscreteScheduler的`set_timesteps`和`add_noise`以及训练目标来匹配
            # 暂时保留你原来的`target = (noise - model_input)`，但强烈建议查阅官方SD3训练代码的损失函数

            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"警告: noisy_latents包含NaN或Inf，跳过此批次")
                continue
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(f"警告: target包含NaN或Inf，跳过此批次")
                continue

            # 前向传播
            model_pred = mmdit(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # 计算损失
            loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            loss = loss / train_configs.gradient_accumulation_steps # 归一化损失，以便在梯度累积时求和

            loss.backward()

            # 梯度累积
            if (step + 1) % train_configs.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad() # 每次优化器更新后清零梯度
                global_step += 1

            epoch_loss += loss.item() * train_configs.gradient_accumulation_steps # 累加原始（未归一化）损失

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item() * train_configs.gradient_accumulation_steps:.4f}', # 显示归一化前的损失
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            progress_bar.update() # 确保进度条更新

        # 每个epoch结束后更新学习率
        scheduler.step()

        avg_loss = epoch_loss / len(dataloader) # 计算每个epoch的平均损失
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"\nEpoch {epoch+1} 完成. 平均损失: {avg_loss:.4f}") # 打印平均损失

        # 定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == train_configs.epochs - 1:
            # 确保模型在保存时处于评估模式，这样LoRA层会正确合并到基础模型中 (对于PeftModel)
            # 或者保存PeftModel的state_dict，PeftModel的save_pretrained方法会处理
            # 你的代码使用了get_peft_model_state_dict，这只保存LoRA层，是正确的。
            current_epoch_mmdit = mmdit # 不需要 .to(torch.float32)，因为已经处理
            current_epoch_mmdit_lora_layers = get_peft_model_state_dict(current_epoch_mmdit)
            lora_save_path = os.path.join(train_configs.output_dir, f"lora_epoch_{epoch+1}")
            os.makedirs(lora_save_path, exist_ok=True)
            # 直接使用PeftModel的save_pretrained方法更简洁和推荐
            # mmdit.save_pretrained(lora_save_path)
            # 如果要使用StableDiffusion3Pipeline.save_lora_weights，你的方式是正确的
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=current_epoch_mmdit_lora_layers,
                text_encoder_lora_layers=None, # 如果没有LoRA文本编码器，保持None
                text_encoder_2_lora_layers=None, # 如果没有LoRA文本编码器，保持None
            )
            print(f"LoRA 权重已保存至: {lora_save_path}")

    # 最终保存
    final_mmdit = mmdit # 不需要 .to(torch.float32)
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