import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer # BitsAndBytesConfig was imported but not used
import json # For saving metadata (optional)

# --- 配置参数 ---
model_path = "/root/autodl-tmp/models/MiniCPM-V-2_6"
image_dir = "/root/autodl-tmp/HuggingFace_Datasets/Nechintosh--ghibli/Nechintosh_ghibli/"
# 输出 captions 的目录，建议与图片目录分开或在其子目录中
caption_output_dir = os.path.join("/root/autodl-tmp/HuggingFace_Datasets/Nechintosh--ghibli/", "captions_gstyle") # 将txt文件保存在原图片目录下的 captions_gstyle 子目录
# 如果你希望 captions 和图片放在同一个文件夹，可以直接用 image_dir
# caption_output_dir = image_dir

# 创建输出目录 (如果不存在)
os.makedirs(caption_output_dir, exist_ok=True)

# 支持的图片扩展名
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')

question = "This is a Ghibli-style image. Please give descriptive synthetic captions for the image for LoRA fine-tuning."
# 或者更简洁的，如果模型能够理解上下文：
# question = "A Ghibli-style image. Descriptive synthetic caption for LoRA fine-tuning:"

# --- 加载模型和 Tokenizer (只需要加载一次) ---
print("正在加载模型和Tokenizer...")
try:
    # 注意：如果你的显存有限，可以考虑 BitsAndBytesConfig 进行量化
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        # attn_implementation='sdpa', # or 'flash_attention_2' if supported and installed
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config # 如果使用量化，取消注释这一行
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("模型和Tokenizer加载完成。")
except Exception as e:
    print(f"加载模型或Tokenizer时发生错误: {e}")
    exit()

# --- 遍历图片并生成描述 ---
all_captions_metadata = [] # 用于存储所有图片的元数据 (可选)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
total_images = len(image_files)
print(f"共找到 {total_images} 张图片待处理。")

for i, image_filename in enumerate(image_files):
    image_path = os.path.join(image_dir, image_filename)
    base_filename, _ = os.path.splitext(image_filename)
    caption_txt_path = os.path.join(caption_output_dir, f"{base_filename}.txt")

    print(f"正在处理第 {i+1}/{total_images} 张图片: {image_filename} ...")

    # 如果caption文件已存在，可以选择跳过
    # if os.path.exists(caption_txt_path):
    #     print(f"Caption 文件 {caption_txt_path} 已存在，跳过。")
    #     # (可选) 如果需要，可以读取已存在的caption并存入metadata
    #     # with open(caption_txt_path, 'r', encoding='utf-8') as f_cap:
    #     #     existing_caption = f_cap.read().strip()
    #     # all_captions_metadata.append({"image": image_filename, "caption": existing_caption})
    #     continue

    try:
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question]}]

        # 模型推理
        res = model.chat(
            image=None, # MiniCPM-V 的 chat API 似乎是将 image 放在 msgs 里
            msgs=msgs,
            tokenizer=tokenizer,
            # 可以根据需要调整模型的生成参数
            # sampling=True,
            # temperature=0.7,
            # top_p=0.8,
            # repetition_penalty=1.05,
        )
        
        # 假设 res 直接是字符串类型的 caption
        # 如果 res 是一个包含更复杂结构的对象，你需要从中提取文本，例如:
        # caption_text = res['choices'][0]['message']['content'] # 示例，具体结构取决于模型输出
        caption_text = str(res).strip() # 确保是字符串并去除首尾空白

        if not caption_text:
            print(f"警告: 为 {image_filename} 生成的caption为空。")
            caption_text = "ghibli style" # 提供一个默认的占位符，或根据需求处理

        # 1. 保存为与图片同名的 .txt 文件 (LoRA 训练常用)
        with open(caption_txt_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        print(f"Caption 已保存到: {caption_txt_path}")

        # 2. (可选) 收集元数据，以便后续保存为单个JSON文件
        all_captions_metadata.append({
            "image_filename": image_filename, # 原图片文件名
            "caption_file": os.path.basename(caption_txt_path), # caption文件名
            "caption": caption_text
        })

    except FileNotFoundError:
        print(f"错误: 图片文件未找到 {image_path}")
    except AttributeError as ae:
        # 捕捉模型输出不是预期类型的情况
        print(f"错误: 处理图片 {image_filename} 时模型输出解析失败。原始输出: {res}。错误: {ae}")
        print("请检查模型输出 `res` 的实际结构，并相应地提取caption文本。")
    except Exception as e:
        print(f"处理图片 {image_filename} 时发生错误: {e}")
        # 可以在这里决定是跳过该图片还是中止脚本
        # continue

# --- (可选) 保存所有 captions到一个JSON元数据文件 ---
if all_captions_metadata:
    metadata_json_path = os.path.join(caption_output_dir, "_metadata_captions.jsonl") # 使用 .jsonl 格式，每行一个JSON对象
    try:
        with open(metadata_json_path, 'w', encoding='utf-8') as f_meta:
            for item in all_captions_metadata:
                f_meta.write(json.dumps(item) + '\n')
        print(f"所有 captions 的元数据已保存到: {metadata_json_path}")
    except Exception as e:
        print(f"保存元数据文件失败: {e}")

print("所有图片处理完毕！")