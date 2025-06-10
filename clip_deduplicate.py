import os
from PIL import Image
import torch
from tqdm import tqdm
from shutil import copyfile
from transformers import CLIPModel, CLIPProcessor

# ========================
# 参数设置
# ========================
input_folder = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered"  # 原始连续帧文件夹
output_folder = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered-deduplicate"  # 去重后的输出文件夹
similarity_threshold = 0.90  # 相似度阈值，越高越严格
max_images = None  # 限制最大处理图片数（None = 不限制）

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

# ========================
# 初始化 CLIP 模型 - 使用 Hugging Face
# ========================
# 选项1：使用在线模型（自动下载）
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 选项2：如果你有本地模型，取消下面两行的注释并注释上面两行
model = CLIPModel.from_pretrained("/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32")

model = model.to(device)
model.eval()

# ========================
# 提取图像 embedding
# ========================
def get_clip_embedding(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        # 使用 Hugging Face 的处理方式
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 使用 Hugging Face CLIP 模型获取图像特征
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# ========================
# 去重主逻辑
# ========================
def deduplicate_images(input_folder, output_folder, threshold=0.95, max_images=None):
    os.makedirs(output_folder, exist_ok=True)
    embeddings = []
    saved_paths = []

    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
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
            # 复制文件到输出文件夹
            dest_path = os.path.join(output_folder, os.path.basename(img_path))
            copyfile(img_path, dest_path)

    print(f"Kept {len(saved_paths)} unique images out of {len(image_files)}.")
    print(f"Saved to: {output_folder}")

# ========================
# 执行
# ========================
if __name__ == "__main__":
    deduplicate_images(input_folder, output_folder, threshold=similarity_threshold, max_images=max_images)