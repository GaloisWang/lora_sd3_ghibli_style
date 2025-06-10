# filter_dataset.py

import os
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from shutil import copyfile

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device:{device}")


# 直接从本地路径加载 Hugging Face 格式的模型
model = CLIPModel.from_pretrained("/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32")
preprocess = CLIPProcessor.from_pretrained("/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32")

model = model.to(device)
model.eval()


# 获取图像embedding - 修正版本
def get_clip_embeddings(folder, max_imgs=None):
    paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if max_imgs:
        paths = paths[:max_imgs]

    embeddings = []
    for path in tqdm(paths, desc=f"Embedding {folder}"):
        try:
            img = Image.open(path).convert("RGB")
            # 使用 Hugging Face 的方式处理图像
            inputs = preprocess(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 使用 Hugging Face CLIP 模型的正确方法
                image_features = model.get_image_features(**inputs)
                # 归一化
                feat = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append((path, feat.squeeze(0)))
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return embeddings

# 筛选函数
def filter_similar_images(a_embeddings, b_embeddings, threshold=0.95):
    retained = []
    for a_path, a_vec in tqdm(a_embeddings, desc="Filtering A"):
        max_sim = 0
        for _, b_vec in b_embeddings:
            sim = torch.dot(a_vec, b_vec).item()
            if sim > max_sim:
                max_sim = sim
            if max_sim >= threshold:
                break
        if max_sim < threshold:
            retained.append(a_path)
    return retained

def NetTurbo():
    import subprocess
    import os

    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value


if __name__ == "__main__":
    NetTurbo()

    # "https://hf-mirror.com"
    # 'hf-cdn.sufy.com'
    os.environ['HF_ENDPOINT'] = "hf-cdn.sufy.com"

    dataset_a = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli/ItzLoghotXD_Ghibli/my-neighbor-totoro/"
    dataset_b = "/root/autodl-tmp/HuggingFace_Datasets/Nechintosh--ghibli/Nechintosh_ghibli"
    output_folder = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered"  # 最终 real_style_folder

    os.makedirs(output_folder, exist_ok=True)

    emb_A = get_clip_embeddings(dataset_a)
    emb_B = get_clip_embeddings(dataset_b)

    retained_paths = filter_similar_images(emb_A, emb_B, threshold=0.85)
    print(f"Retained {len(retained_paths)} out of {len(emb_A)} images.")

    for path in retained_paths:
        fname = os.path.basename(path)
        copyfile(path, os.path.join(output_folder, fname))

    print("Filtering complete.")