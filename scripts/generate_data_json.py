import json
import os
from pathlib import Path
import argparse


def check_directories(image_dir, captions_dir):
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        return False
    if not os.path.exists(captions_dir):
        print(f"错误: 字幕目录 {captions_dir} 不存在")
        return False
    return True


def generate_json(image_dir, captions_dir, output_path):
    data = []

    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    image_files = [
        f
        for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    print(f"找到 {len(image_files)} 张图片")

    for image_file in image_files:

        image_name = os.path.splitext(image_file)[0]

        caption_filename = f"{image_name}.txt"
        caption_path = os.path.join(captions_dir, caption_filename)

        if not os.path.exists(caption_path):
            print(f"警告: 未找到对应字幕文件 {caption_filename}，跳过")
            continue

        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                caption_text = f.read().strip()
        except Exception as e:
            print(f"错误: 读取字幕文件 {caption_path} 失败: {e}")
            continue

        data.append({"image": image_file, "text": caption_text})

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"成功生成JSON文件: {output_path}")
        print(f"共处理 {len(data)} 对图片和字幕")
    except Exception as e:
        print(f"错误: 写入JSON文件失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="生成图片和字幕对应的JSON文件")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/root/autodl-tmp/HuggingFace_Datasets/111/images/",
        help="图片文件夹路径",
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="/root/autodl-tmp/HuggingFace_Datasets/111/captions/",
        help="字幕文件夹路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/root/autodl-tmp/HuggingFace_Datasets/111/data.json",
        help="输出JSON文件路径",
    )

    args = parser.parse_args()

    if not check_directories(args.image_dir, args.captions_dir):
        return

    generate_json(args.image_dir, args.captions_dir, args.output_path)


if __name__ == "__main__":
    main()
