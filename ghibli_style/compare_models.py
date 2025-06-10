import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
from datetime import datetime
import os
import logging
import gc


# ==== 日志配置 ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pipeline(model_path, lora_path=None):
    print(f"Loading base model from: {model_path}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")

    if lora_path:
        print(f"Applying LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        print("No LoRA applied.")

    return pipe


def clear_model(pipe: StableDiffusion3Pipeline):
    """完全清除模型释放显存"""
    logger.info("🧹 Clearing model from memory...")
    del pipe
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 获取清理后的显存信息
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"📊 GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    gc.collect()
    logger.info("✅ Memory cleanup completed")

def generate_image(pipe, prompt, seed=42, width=1024, height=1024, steps=50, guidance=4):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance
    ).images[0]
    return image


def save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    print(f"Image saved to {output_path}")

def run_generation(model_path, prompt, save_path, seed=42, lora_path=None):
    pipe = load_pipeline(model_path, lora_path)
    image = generate_image(pipe, prompt, seed=seed)
    save_image(image, save_path)
    del pipe
    torch.cuda.empty_cache()
    gc.collect()



if __name__ == "__main__":
    # 配置部分
    prompt = (
        "cute style, A beautiful anime-style girl with dark, flowing hair adorned with small flowers, "
        "wearing a traditional light green kimono with floral patterns. She is joyfully holding a dandelion puff "
        "and smiling with her eyes closed, basking in the warm, golden sunlight. The background is a dreamy sky "
        "with soft, fluffy clouds and flying petals, creating a sense of happiness and freedom."
    )
    prompt = "For LoRA fine-tuning, the descriptive synthetic captions for this Ghibli-style image could be: A Journey Awaits: A character embarks on a road trip in a vintage blue truck, loaded with personal belongings and ready to traverse uncharted paths. The Art of Adventure: Amidst a serene countryside backdrop, a traveler's vehicle is packed with essentials, hinting at stories untold and destinations yet to be discovered. Rustic Charm: The nostalgic allure of an old-fashioned truck brimming with life's necessities, set against a picturesque rural landscape that speaks volumes about simplicity and exploration. These captions aim to capture the essence of the scene while also providing potential narrative elements that can be incorporated into a LoRA model for more expressive and contextually rich outputs."

    seed = 1641421826
    output_dir = "/root/Codes/lora_diffusion/ghibli_style/ghibli_style_output_images"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    guidance = 7.5

    # 模型路径
    official_model_path = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    lora_model_path = "/home/lora_sd3_train_logs_fulldata_060216e5/lora_final//"
    segments = lora_model_path.strip(os.sep).split(os.sep)  # 分割路径为列表
    target_segment = next((s for s in segments if "lora_sd3_" in s), None)

    os.makedirs(output_dir,exist_ok=True)

    # run_generation(
    #     official_model_path,
    #     prompt,
    #     f"{output_dir}/origin_{target_segment}_cfg{guidance}_{timestamp}.png",
    #     seed=seed,
    #     lora_path=None,
    # )

    run_generation(
        official_model_path,
        prompt,
        f"{output_dir}/copax_lora_{target_segment}_cfg{guidance}_{timestamp}.png",
        seed=seed,
        lora_path=lora_model_path,
    )

