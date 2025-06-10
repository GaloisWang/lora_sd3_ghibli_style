# test.py
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig

model_path = "/root/autodl-tmp/models/MiniCPM-V-2_6"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

image_dir = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli/ItzLoghotXD_Ghibli/my-neighbor-totoro/"
image = Image.open(os.path.join(image_dir,'00001.jpg')).convert('RGB')
# question = 'This is a Ghibli-style picture. Please give descriptive synthetic captions for the picture.'
question = "This is a Ghibli-style image. Please give descriptive synthetic captions for the image for LoRA fine-tuning."
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print("模型答案是:{0}".format(res))
