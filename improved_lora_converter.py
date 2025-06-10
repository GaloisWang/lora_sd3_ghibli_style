import os
import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any, Optional, List
import re

class ImprovedSD3LoRAConverter:
    def __init__(self):
        # SD3的标准目标模块
        self.sd3_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out", 
            "ff.net.0.proj", "ff.net.2"
        ]
        
        # 键名映射规则
        self.key_mappings = {
            # 移除常见前缀
            r'^lora_unet_': '',
            r'^lora_te_': 'text_encoder.',
            r'^lora_te1_': 'text_encoder.',
            r'^lora_te2_': 'text_encoder_2.',
            r'^lora_te3_': 'text_encoder_3.',
            
            # 标准化层名
            r'_transformer_blocks_': '.transformer_blocks.',
            r'_attn_': '.attn.',
            r'_norm_': '.norm.',
            r'_ff_': '.ff.',
            r'_net_': '.net.',
            
            # 处理下划线到点的转换
            r'transformer_blocks_(\d+)_': r'transformer.transformer_blocks.\1.',
            r'x_embedder_': 'transformer.x_embedder.',
            r'y_embedder_': 'transformer.y_embedder.',
            r'context_embedder_': 'transformer.context_embedder.',
            r't_embedder_': 'transformer.t_embedder.',
            r'norm_out_': 'transformer.norm_out.',
            r'proj_out_': 'transformer.proj_out.',
        }
    
    def normalize_key_name(self, key: str) -> str:
        """标准化权重键名"""
        normalized = key
        
        # 应用映射规则
        for pattern, replacement in self.key_mappings.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # 确保transformer路径正确
        if 'transformer_blocks' in normalized and not normalized.startswith('transformer.'):
            normalized = 'transformer.' + normalized
        
        # 清理多余的点和下划线
        normalized = re.sub(r'\.+', '.', normalized)  # 多个点变一个
        normalized = re.sub(r'_+', '_', normalized)   # 多个下划线变一个
        normalized = normalized.strip('._')           # 移除首尾的点和下划线
        
        return normalized
    
    def analyze_original_lora(self, lora_path: str) -> Dict[str, Any]:
        """分析原始LoRA文件"""
        analysis = {
            "files": [],
            "tensors": {},
            "statistics": {
                "total_tensors": 0,
                "lora_pairs": 0,
                "alpha_tensors": 0,
                "unique_modules": set()
            }
        }
        
        lora_dir = Path(lora_path)
        safetensor_files = list(lora_dir.glob("*.safetensors"))
        
        for file_path in safetensor_files:
            analysis["files"].append(str(file_path))
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    analysis["tensors"][key] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "file": str(file_path),
                        "normalized_key": self.normalize_key_name(key)
                    }
                    
                    analysis["statistics"]["total_tensors"] += 1
                    
                    # 统计LoRA组件
                    if "lora_A" in key or "lora_a" in key:
                        analysis["statistics"]["lora_pairs"] += 0.5
                    elif "lora_B" in key or "lora_b" in key:
                        analysis["statistics"]["lora_pairs"] += 0.5
                    elif "alpha" in key.lower():
                        analysis["statistics"]["alpha_tensors"] += 1
                    
                    # 提取模块名
                    module_name = self.extract_module_name(key)
                    analysis["statistics"]["unique_modules"].add(module_name)
        
        analysis["statistics"]["unique_modules"] = list(analysis["statistics"]["unique_modules"])
        return analysis
    
    def extract_module_name(self, key: str) -> str:
        """从键名中提取模块名"""
        # 移除LoRA特定后缀
        clean_key = re.sub(r'\.(lora_[AB]|alpha)$', '', key, flags=re.IGNORECASE)
        
        # 提取最后的模块名
        parts = clean_key.split('.')
        if len(parts) >= 2:
            return parts[-1]  # 返回最后一个部分
        return clean_key
    
    def create_adapter_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建适配器配置"""
        # 从分析结果中确定目标模块
        detected_modules = []
        for module in analysis["statistics"]["unique_modules"]:
            if module in self.sd3_target_modules:
                detected_modules.append(module)
        
        # 如果没有检测到标准模块，使用分析出的所有模块
        if not detected_modules:
            detected_modules = list(analysis["statistics"]["unique_modules"])
        
        config = {
            "base_model_name_or_path": "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/",
            "library_name": "diffusers",
            "peft_type": "LORA",
            "target_modules": detected_modules,
            "task_type": "DIFFUSION",
            "inference_mode": False,
            "r": 16,  # 默认rank
            "lora_alpha": 16,  # 默认alpha
            "lora_dropout": 0.0,
            "bias": "none",
            "use_rslora": False,
            "use_dora": False
        }
        
        return config
    
    def convert_lora(self, input_path: str, output_path: str, 
                    force_overwrite: bool = False) -> bool:
        """转换LoRA模型"""
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        
        if output_dir.exists() and not force_overwrite:
            print(f"输出目录已存在: {output_dir}")
            print("使用 force_overwrite=True 来覆盖")
            return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 分析原始LoRA文件...")
        analysis = self.analyze_original_lora(str(input_dir))
        
        print(f"发现 {analysis['statistics']['total_tensors']} 个张量")
        print(f"发现 {int(analysis['statistics']['lora_pairs'])} 个LoRA对")
        print(f"目标模块: {analysis['statistics']['unique_modules']}")
        
        # 转换张量
        print("\n🔄 转换张量...")
        converted_tensors = {}
        conversion_log = []
        
        for old_key, tensor_info in analysis["tensors"].items():
            new_key = tensor_info["normalized_key"]
            
            # 加载张量
            with safe_open(tensor_info["file"], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(old_key)
            
            converted_tensors[new_key] = tensor
            conversion_log.append(f"{old_key} -> {new_key}")
            print(f"  {old_key} -> {new_key}")
        
        # 保存转换后的张量
        adapter_file = output_dir / "adapter_model.safetensors"
        save_file(converted_tensors, adapter_file)
        print(f"✅ 保存权重文件: {adapter_file}")
        
        # 创建配置文件
        config = self.create_adapter_config(analysis)
        config_file = output_dir / "adapter_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存配置文件: {config_file}")
        
        # 保存转换日志
        log_file = output_dir / "conversion_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("LoRA转换日志\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"输入路径: {input_path}\n")
            f.write(f"输出路径: {output_path}\n")
            f.write(f"总张量数: {len(converted_tensors)}\n\n")
            f.write("键名转换:\n")
            for log_entry in conversion_log:
                f.write(f"  {log_entry}\n")
        
        print(f"✅ 保存转换日志: {log_file}")
        
        return True
    
    def validate_converted_lora(self, lora_path: str) -> Dict[str, Any]:
        """验证转换后的LoRA"""
        validation = {
            "valid": False,
            "issues": [],
            "tensor_count": 0,
            "lora_pairs": 0
        }
        
        lora_dir = Path(lora_path)
        
        # 检查必需文件
        adapter_file = lora_dir / "adapter_model.safetensors"
        config_file = lora_dir / "adapter_config.json"
        
        if not adapter_file.exists():
            validation["issues"].append("缺少adapter_model.safetensors")
            return validation
        
        if not config_file.exists():
            validation["issues"].append("缺少adapter_config.json")
            return validation
        
        # 验证配置文件
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            required_keys = ["peft_type", "target_modules", "task_type"]
            for key in required_keys:
                if key not in config:
                    validation["issues"].append(f"配置文件缺少: {key}")
        except Exception as e:
            validation["issues"].append(f"配置文件读取错误: {str(e)}")
        
        # 验证权重文件
        try:
            with safe_open(adapter_file, framework="pt", device="cpu") as f:
                tensor_names = list(f.keys())
                validation["tensor_count"] = len(tensor_names)
                
                # 计算LoRA对
                lora_a_count = sum(1 for name in tensor_names if "lora_A" in name or "lora_a" in name)
                lora_b_count = sum(1 for name in tensor_names if "lora_B" in name or "lora_b" in name)
                validation["lora_pairs"] = min(lora_a_count, lora_b_count)
                
                if validation["lora_pairs"] == 0:
                    validation["issues"].append("未找到有效的LoRA权重对")
        except Exception as e:
            validation["issues"].append(f"权重文件读取错误: {str(e)}")
        
        validation["valid"] = len(validation["issues"]) == 0
        return validation

# 使用示例
def main():
    converter = ImprovedSD3LoRAConverter()
    
    # 配置路径
    input_lora = "/home/lora_sd3_train_logs_0524_1600_0525_1600/lora_final_converted/"
    output_lora = "/home/lora_sd3_train_logs_0524_1600_0525_1600/lora_final_converted2/"
    
    print("🚀 开始改进的LoRA转换...")
    
    # 转换LoRA
    success = converter.convert_lora(
        input_path=input_lora,
        output_path=output_lora,
        force_overwrite=True
    )
    
    if success:
        print("\n✅ 转换完成!")
        
        # 验证转换结果
        print("\n🔍 验证转换结果...")
        validation = converter.validate_converted_lora(output_lora)
        
        if validation["valid"]:
            print("🎉 验证通过!")
            print(f"张量数量: {validation['tensor_count']}")
            print(f"LoRA对数: {validation['lora_pairs']}")
        else:
            print("❌ 验证失败:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
    else:
        print("❌ 转换失败")
    
    return output_lora if success else None

if __name__ == "__main__":
    converted_path = main()
    if converted_path:
        print(f"\n📁 转换后的LoRA路径: {converted_path}")
        print("现在可以用这个路径测试LoRA加载了!")