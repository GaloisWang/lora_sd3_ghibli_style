import torch
from diffusers import StableDiffusion3Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

official_model_path = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
lora_model_dir =  "/home/models/lpyy-trained-sd3-lora/"

# 1. 加载官方模型（到CPU以节省显存）
logger.info("加载官方模型到CPU...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    official_model_path,
    torch_dtype=torch.float16, # 建议也用fp16加载以匹配后续操作
    low_cpu_mem_usage=True,
    variant="fp16" # 确保加载fp16变体
).to("cpu")
logger.info("官方模型加载完成。")


# 2. 记录原始模型特定层的权重
# 注意：SD3使用DiT架构，其层访问路径是 transformer_blocks
try:
    original_weight = pipe.transformer.transformer_blocks[0].attn.to_q.weight.data.clone()
    logger.info(f"成功获取原始模型 to_q 层的权重，形状: {original_weight.shape}")
except AttributeError as e:
    logger.error(f"访问 transformer 层时发生错误: {e}")
    logger.error("请检查 pipe.transformer 的属性，可以使用 dir(pipe.transformer) 查看。")
    # 可以在这里打印 dir(pipe.transformer) 来查看所有可用属性
    # print(dir(pipe.transformer))


# 3. 加载LoRA权重
logger.info(f"加载LoRA权重：{lora_model_dir}")
try:
    pipe.load_lora_weights(lora_model_dir)
    logger.info("LoRA权重加载完成。")
except Exception as e:
    logger.error(f"加载LoRA权重时发生错误: {e}")
    logger.error("这可能是由于PEFT保存的LoRA与Diffusers加载格式不兼容导致的。")


# 4. 记录加载LoRA后相同层的权重
try:
    lora_loaded_weight = pipe.transformer.transformer_blocks[0].attn.to_q.weight.data.clone()
    logger.info(f"成功获取加载LoRA后 to_q 层的权重，形状: {lora_loaded_weight.shape}")
except AttributeError as e:
    logger.error(f"访问 transformer 层时发生错误（加载LoRA后）: {e}")


# 5. 比较两个权重的差异
if 'original_weight' in locals() and 'lora_loaded_weight' in locals():
    difference = (original_weight - lora_loaded_weight).abs().sum()
    if difference > 1e-6: # 使用一个小的阈值来判断是否存在差异
        logger.info(f"🎉 原始权重与加载LoRA后的权重存在差异！总绝对差异: {difference.item():.6f}")
        logger.info("这表明LoRA权重已被成功应用并修改了模型参数。")
    else:
        logger.warning(f"⚠️ 原始权重与加载LoRA后的权重几乎相同。总绝对差异: {difference.item():.6f}")
        logger.warning("这可能意味着LoRA权重未被正确应用，或者它们对模型参数的影响微乎其微。")
else:
    logger.error("无法进行权重比较，因为未能成功获取原始或加载LoRA后的权重。")