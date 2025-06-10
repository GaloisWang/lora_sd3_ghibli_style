from transformers import AutoTokenizer, AutoModel, CLIPProcessor
import torch
import os

# Define the base directory where you downloaded the Hugging Face model
local_hf_model_dir = "/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32/"

# Check if the directory exists
if not os.path.isdir(local_hf_model_dir):
    raise FileNotFoundError(f"CLIP模型本地目录未找到: {local_hf_model_dir}。请检查路径。")

try:
    clip_model = AutoModel.from_pretrained(local_hf_model_dir)
    clip_tokenizer = AutoTokenizer.from_pretrained(local_hf_model_dir)
    clip_processor = CLIPProcessor.from_pretrained(local_hf_model_dir)
    
except Exception as e:
    print(f"Error loading Hugging Face CLIP model from {local_hf_model_dir}: {e}")
    print("Please ensure the directory contains all necessary files (e.g., config.json, pytorch_model.bin, tokenizer_config.json, vocab.json, merges.txt, preprocessor_config.json).")
    raise

# 将模型移动到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)


text_input = ["a photo of a cat", "a photo of a dog"]
tokenized_text = clip_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
text_features = clip_model.get_text_features(**tokenized_text)
print("Text features shape:", text_features.shape)



print(f"CLIP模型已从本地Hugging Face格式目录加载: {local_hf_model_dir}")