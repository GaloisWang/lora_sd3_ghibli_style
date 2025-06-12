import os
from PIL import Image
import torch
from tqdm import tqdm
from shutil import copyfile
from transformers import CLIPModel, CLIPProcessor


input_folder = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered"
output_folder = (
    "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered-deduplicate"
)
similarity_threshold = 0.90
max_images = None

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")


model = CLIPModel.from_pretrained(
    "/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32"
)
processor = CLIPProcessor.from_pretrained(
    "/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32"
)

model = model.to(device)
model.eval()


def get_clip_embedding(img_path):
    try:
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def deduplicate_images(input_folder, output_folder, threshold=0.95, max_images=None):
    os.makedirs(output_folder, exist_ok=True)
    embeddings = []
    saved_paths = []

    image_files = sorted(
        [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files, desc="Deduplicating"):
        embedding = get_clip_embedding(img_path)
        if embedding is None:
            continue

        is_duplicate = False
        for ref in embeddings:
            sim = torch.dot(embedding, ref).item()
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            embeddings.append(embedding)
            saved_paths.append(img_path)

            dest_path = os.path.join(output_folder, os.path.basename(img_path))
            copyfile(img_path, dest_path)

    print(f"Kept {len(saved_paths)} unique images out of {len(image_files)}.")
    print(f"Saved to: {output_folder}")


if __name__ == "__main__":
    deduplicate_images(
        input_folder,
        output_folder,
        threshold=similarity_threshold,
        max_images=max_images,
    )
