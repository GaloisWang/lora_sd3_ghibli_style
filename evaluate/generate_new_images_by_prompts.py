import os
import gc
import torch
import logging
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pipeline(model_path: str, lora_path: str = None) -> StableDiffusion3Pipeline:
    logger.info(f"Loading base model from: {model_path}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    if lora_path:
        logger.info(f"Applying LoRA weights from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        logger.info("No LoRA weights applied.")

    return pipe


def clear_model(pipe: StableDiffusion3Pipeline):
    logger.info("Clearing model from memory...")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        )
    gc.collect()
    logger.info("Memory cleanup completed")


def generate_image(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    guidance: float = 7.5,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance,
    )
    return result.images[0]


def save_image(image, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    logger.info(f"Image saved to: {path}")


def run_generation(
    model_path, prompt, save_path, seed=42, lora_path=None, guidance=7.5
):
    pipe = load_pipeline(model_path, lora_path)
    image = generate_image(pipe, prompt, seed=seed, guidance=guidance)
    save_image(image, save_path)
    clear_model(pipe)


if __name__ == "__main__":
    prompt = (
        "For LoRA fine-tuning, the descriptive synthetic captions for this Ghibli-style image could be: "
        "A Journey Awaits: A character embarks on a road trip in a vintage blue truck, loaded with personal belongings and ready to traverse uncharted paths. "
        "The Art of Adventure: Amidst a serene countryside backdrop, a traveler's vehicle is packed with essentials, hinting at stories untold and destinations yet to be discovered. "
        "Rustic Charm: The nostalgic allure of an old-fashioned truck brimming with life's necessities, set against a picturesque rural landscape that speaks volumes about simplicity and exploration."
    )

    seed = 1641421826
    guidance = 7.5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(
        "/root/Codes/lora_diffusion/ghibli_style/ghibli_style_output_images"
    )
    model_path = (
        "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    )
    lora_path = "/home/lora_sd3_train_logs_fulldata_060216e5/lora_final/"

    target_segment = next(
        (s for s in Path(lora_path).parts if "lora_sd3_" in s), "custom_lora"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base Model
    run_generation(
        model_path,
        prompt,
        output_dir / f"origin_{target_segment}_cfg{guidance}_{timestamp}.png",
        seed=seed,
        lora_path=None,
        guidance=guidance,
    )

    # 2. LoRA Fine-Tuned Model
    run_generation(
        model_path,
        prompt,
        output_dir / f"copax_lora_{target_segment}_cfg{guidance}_{timestamp}.png",
        seed=seed,
        lora_path=lora_path,
        guidance=guidance,
    )
