import torch
from diffusers import StableDiffusion3Pipeline as DiffusionPipeline

official_model_path = (
    "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
)
lora_model_dir = "/home/models/copax_cute_style_sd3/"

prompt = "cute style, A beautiful anime-style girl with dark, flowing hair adorned with small flowers, wearing a traditional light green kimono with floral patterns. She is joyfully holding a dandelion puff and smiling with her eyes closed, basking in the warm, golden sunlight. The background is a dreamy sky with soft, fluffy clouds and flying petals, creating a sense of happiness and freedom."

pipeline = DiffusionPipeline.from_pretrained(
    official_model_path, torch_dtype=torch.float16
).to("cuda")
# pipeline.load_lora_weights(lora_model_dir)
image = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    generator=torch.Generator(device="cuda").manual_seed(1641421826),
    width=1024,
    height=1024,
    guidance_scale=4,
).images[0]
image.save("ff7_output_origin_cfg4.png", format="PNG")
