import os
import subprocess
import json
from dataclasses import dataclass, asdict
from typing import Optional
from omegaconf import OmegaConf

@dataclass
class TrainParametersConfig:
    image_dir: str
    captions_dir: str
    output_dir: str
    epochs: int = 50
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    lr_scheduler: str = "constant" 
    rank: int = 16
    lora_alpha: int = 16
    pretrained_model_path: str = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    resolution: int = 1024

def load_training_config(config_path: str) -> TrainParametersConfig:
    """从json文件中加载训练参数配置"""
    if config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
    else:
        data_dict = OmegaConf.load(config_path)
    return TrainParametersConfig(**data_dict)

def train_dreambooth_lora(
    config_path: Optional[str] = "./lora_sd3_config.json",
    mixed_precision: str = "fp16",
    lr_warmup_steps: int = 0,
    checkpointing_steps: int = 500,
    max_train_steps: Optional[int] = None,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = False,
    dataloader_num_workers: int = 0,
    push_to_hub: bool = False,
    # 可选参数 - 如果需要验证功能可以启用
    enable_validation: bool = False,
    instance_prompt: str = "ghibli style",
    validation_prompt: str = "A ghibli style landscape",
    validation_epochs: int = 10,
    # 可选参数 - 如果需要先验保存可以启用  
    enable_prior_preservation: bool = False,
    class_prompt: str = "a photo",
    num_class_images: int = 100,
    **kwargs
):
    
    # 加载配置文件
    try:
        train_configs = load_training_config(config_path)
        print(f"成功加载配置文件: {config_path}")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return
    
    # 检查路径是否存在
    if not os.path.exists(train_configs.image_dir):
        print(f"错误: 图片目录不存在: {train_configs.image_dir}")
        return
    
    if not os.path.exists(train_configs.captions_dir):
        print(f"错误: 标注目录不存在: {train_configs.captions_dir}")
        return
    
    if not os.path.exists(train_configs.pretrained_model_path):
        print(f"错误: 预训练模型路径不存在: {train_configs.pretrained_model_path}")
        return
    
    # 创建输出目录
    os.makedirs(train_configs.output_dir, exist_ok=True)
    
    # 配置环境变量
    os.environ["OUTPUT_DIR"] = train_configs.output_dir
    
    # 构建训练脚本路径 (假设在当前目录下载了diffusers仓库)
    script_path = "/root/Codes/diffusers-0.33.1/examples/dreambooth/train_dreambooth_lora_sd3.py"
    
    # 构建基础命令参数
    cmd = [
        "accelerate", "launch", script_path,
        f"--pretrained_model_name_or_path={train_configs.pretrained_model_path}",
        f"--instance_data_dir={train_configs.image_dir}",
        f"--output_dir={train_configs.output_dir}",
        f"--mixed_precision={mixed_precision}",
        f"--resolution={train_configs.resolution}",
        f"--train_batch_size={train_configs.train_batch_size}",
        f"--learning_rate={train_configs.learning_rate}",
        f"--lr_scheduler={train_configs.lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--checkpointing_steps={checkpointing_steps}",
        f"--rank={train_configs.rank}",
        f"--dataloader_num_workers={dataloader_num_workers}",
    ]
    
    cmd.extend([
        f"--instance_prompt={instance_prompt}",
    ])
    
    # 可选：启用验证功能
    if enable_validation:
        cmd.extend([
            f"--validation_prompt={validation_prompt}",
            f"--validation_epochs={validation_epochs}",
        ])
    
    # 可选：启用先验保存功能
    if enable_prior_preservation:
        cmd.extend([
            "--with_prior_preservation",
            "--class_data_dir=/tmp/class_images",
            f"--class_prompt={class_prompt}",
            f"--num_class_images={num_class_images}",
        ])
    

    cmd.append(f"--num_train_epochs={train_configs.epochs}")
    
    # 可选参数
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    

    
    # 针对3090显卡的优化设置
    cmd.extend([
        "--gradient_checkpointing",  # 节省显存
        "--train_text_encoder",  # 训练文本编码器
        "--sample_batch_size=1",  # 验证时的批次大小
        "--max_grad_norm=1.0",  # 梯度裁剪
    ])
    
    # 打印完整命令用于调试
    print("执行命令:")
    print(" ".join(cmd))
    print("-" * 80)
    
    # 执行训练命令
    try:
        print(f"开始训练...")
        print(f"配置文件: {config_path}")
        print(f"图片目录: {train_configs.image_dir}")
        print(f"标注目录: {train_configs.captions_dir}")
        print(f"输出目录: {train_configs.output_dir}")
        print(f"训练轮数: {train_configs.epochs}")
        print(f"学习率: {train_configs.learning_rate}")
        print(f"批次大小: {train_configs.train_batch_size}")
        print(f"LoRA rank: {train_configs.rank}")
        print(f"LoRA alpha: {train_configs.lora_alpha}")
        print("-" * 80)
        
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # 实时显示输出
        )
        
        print(f"\n训练完成！模型已保存至: {train_configs.output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"训练失败，错误代码: {e.returncode}")
        print("请检查上方的错误信息")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"发生意外错误: {e}")



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 创建配置文件（如果不存在）
    config_file = "/root/Codes/lora_diffusion/train/lora_sd3_config_subset100.json"
    
    # 开始训练
    train_dreambooth_lora(
        config_path=config_file,
        mixed_precision="fp16",  # 使用fp16节省显存
        lr_warmup_steps=100,  # 添加学习率预热
        checkpointing_steps=250,  # 每250步保存检查点
        gradient_checkpointing=True,  # 启用梯度检查点节省显存
        dataloader_num_workers=0,  # 数据加载线程数
        
        # 可选功能 - 根据需要启用
        enable_validation=False,  # 是否启用验证（会占用显存和时间）
        instance_prompt="ghibli style",  # 简化的提示词
        # validation_prompt="A ghibli style landscape",  # 只在enable_validation=True时需要
        # validation_epochs=5,  # 只在enable_validation=True时需要
        
        enable_prior_preservation=False,  # 是否启用先验保存（风格训练通常不需要）
        # class_prompt="a photo",  # 只在enable_prior_preservation=True时需要
        # num_class_images=100,  # 只在enable_prior_preservation=True时需要
    )