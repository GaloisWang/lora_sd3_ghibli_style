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


def parse_safetensors(file_path):
    # Load the file, load_file returns only the state_dict
    state_dict = load_file(file_path)

    print("-------------------------state_dict---------------------------")
    # Print the state_dict keys and a sample of their shapes/types
    for key, value in state_dict.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Check for metadata. In some safetensors files (e.g., from Hugging Face),
    # metadata might be accessible via a special "__metadata__" key in the state_dict
    # or you might need a separate function like `load_header` if it's external.
    
    metadata = state_dict.get("__metadata__", None) # Common way to get embedded metadata
    
    print("-------------------------metadata-----------------------------")
    if metadata:
        print("Metadata found:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("No embedded metadata found directly in state_dict (or '__metadata__' key not present).")

    # Print the number of parameters
    print(f"\n模型参数数量 (tensors in state_dict): {len(state_dict)}")
    

if __name__ == "__main__":
    lora_paths_dir = "/home/lora_sd3_train_logs_0524_1600_0525_1600/lora_final_converted/"
    lora_paths = find_safetensors_files(lora_paths_dir)
    if len(lora_paths) == 1:
        print("找到lora目录下的权重文件,路径为:{0}".format(lora_paths[0]))
    else:
        print("lora_paths_dir下的权重文件个数不为1")
        exit(0)
    parse_safetensors(lora_paths[0])