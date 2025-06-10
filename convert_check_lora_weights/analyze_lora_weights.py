import os
from safetensors.torch import load_file

def find_safetensors_files(root_dir, recursive=True):
    """
    查找指定目录下的 Safetensors 文件（.safetensors 或 .pt.safetensors 等）
    
    :param root_dir: 根目录路径
    :param recursive: 是否递归查找子目录，默认 True
    :return: 符合条件的文件路径列表
    """
    safetensors_files = []
    # 支持的扩展名（不区分大小写）
    extensions = ('.safetensors', '.SafeTensors', '.SAFETENSORS')
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否匹配
            if filename.lower().endswith(extensions):
                file_path = os.path.join(dirpath, filename)
                safetensors_files.append(file_path)
        # 非递归时仅遍历当前目录
        if not recursive:
            break
    return safetensors_files


def analyze_lora_weights(lora_path):
    print(f"🔍 Loading LoRA weights from: {lora_path}")
    state_dict = load_file(lora_path)

    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    if not lora_keys:
        print("❌ No LoRA layers found in the model.")
        return

    print(f"✅ Found {len(lora_keys)} LoRA layers. Showing statistics:")
    print("-" * 60)
    for name in lora_keys:
        param = state_dict[name]
        mean = param.mean().item()
        std = param.std().item()
        max_val = param.max().item()
        min_val = param.min().item()
        print(f"{name:<50} | mean={mean:.6f} std={std:.6f} max={max_val:.6f} min={min_val:.6f}")
    print("-" * 60)

if __name__ == "__main__":
    lora_paths_dir = "/home/lora_sd3_train_logs_0602_myscript_rank16e5/lora_final/"
    lora_paths = find_safetensors_files(lora_paths_dir)
    if len(lora_paths) == 1:
        print("找到lora目录下的权重文件,路径为:{0}".format(lora_paths[0]))
    else:
        print("lora_paths_dir下的权重文件个数不为1")
        exit(0)

    analyze_lora_weights(lora_paths[0])
