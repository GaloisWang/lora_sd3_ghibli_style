import os
import json
import torch
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import StableDiffusion3Pipeline
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRADiagnosticTool:
    def __init__(self):
        self.sd3_layer_mapping = {
            # Transformer layers mapping
            "transformer.transformer_blocks": "transformer.transformer_blocks",
            "transformer.pos_embed": "transformer.pos_embed", 
            "transformer.x_embedder": "transformer.x_embedder",
            "transformer.y_embedder": "transformer.y_embedder",
            "transformer.context_embedder": "transformer.context_embedder",
            "transformer.t_embedder": "transformer.t_embedder",
            "transformer.norm_out": "transformer.norm_out",
            "transformer.proj_out": "transformer.proj_out"
        }
    
    def diagnose_lora_files(self, lora_dir: str) -> Dict[str, Any]:
        """诊断LoRA文件结构和内容"""
        diagnosis = {
            "files_found": [],
            "total_tensors": 0,
            "lora_layers": [],
            "weight_info": {},
            "issues": []
        }
        
        lora_path = Path(lora_dir)
        if not lora_path.exists():
            diagnosis["issues"].append(f"LoRA目录不存在: {lora_dir}")
            return diagnosis
        
        # 检查文件
        safetensor_files = list(lora_path.glob("*.safetensors"))
        if not safetensor_files:
            diagnosis["issues"].append("未找到.safetensors文件")
            return diagnosis
        
        diagnosis["files_found"] = [str(f) for f in safetensor_files]
        
        # 分析每个文件
        for file_path in safetensor_files:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    tensor_names = f.keys()
                    diagnosis["total_tensors"] += len(tensor_names)
                    
                    for name in tensor_names:
                        tensor = f.get_tensor(name)
                        diagnosis["weight_info"][name] = {
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "non_zero_ratio": (tensor != 0).float().mean().item(),
                            "abs_mean": tensor.abs().mean().item()
                        }
                        
                        # 检查是否为LoRA层
                        if any(lora_key in name.lower() for lora_key in ["lora_a", "lora_b", "alpha"]):
                            diagnosis["lora_layers"].append(name)
                            
            except Exception as e:
                diagnosis["issues"].append(f"读取文件 {file_path} 时出错: {str(e)}")
        
        return diagnosis
    
    def fix_lora_adapter_config(self, lora_dir: str):
        """修复adapter_config.json文件"""
        config_path = Path(lora_dir) / "adapter_config.json"
        
        # 标准的SD3 LoRA配置
        standard_config = {
            "base_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
            "library_name": "diffusers",
            "peft_type": "LORA",
            "target_modules": [
                "to_k",
                "to_q", 
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2"
            ],
            "task_type": "DIFFUSION"
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                print(f"现有配置: {existing_config}")
            except:
                print("无法读取现有配置文件")
        
        # 写入标准配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(standard_config, f, indent=2)
        
        print(f"✅ 已更新adapter_config.json: {config_path}")
    
    def convert_weights_format(self, input_dir: str, output_dir: str):
        """转换权重格式以确保兼容性"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集所有权重
        all_tensors = {}
        
        for safetensor_file in input_path.glob("*.safetensors"):
            print(f"处理文件: {safetensor_file}")
            
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    
                    # 确保权重名称符合Diffusers格式
                    converted_key = self.convert_key_name(key)
                    all_tensors[converted_key] = tensor
                    print(f"  {key} -> {converted_key} | Shape: {tensor.shape}")
        
        if all_tensors:
            # 保存转换后的权重
            output_file = output_path / "adapter_model.safetensors"
            save_file(all_tensors, output_file)
            print(f"✅ 已保存转换后的权重: {output_file}")
            
            # 复制配置文件
            self.fix_lora_adapter_config(str(output_path))
            
            return str(output_path)
        else:
            print("❌ 未找到任何有效的张量")
            return None
    
    def convert_key_name(self, key: str) -> str:
        """转换权重键名以匹配Diffusers格式"""
        # 常见的转换规则
        conversions = {
            # 处理不同的命名约定
            "lora_unet_": "",
            "lora_te_": "text_encoder.",
            "lora_te1_": "text_encoder.",
            "lora_te2_": "text_encoder_2.",
            "lora_te3_": "text_encoder_3.",
        }
        
        converted = key
        for old, new in conversions.items():
            if old in converted:
                converted = converted.replace(old, new)
        
        # 确保transformer层路径正确
        if "transformer_blocks" in converted:
            # 标准化transformer块的路径
            parts = converted.split(".")
            if "transformer_blocks" in parts:
                idx = parts.index("transformer_blocks")
                if idx + 1 < len(parts):
                    # 确保格式为 transformer.transformer_blocks.{num}.{layer}
                    if not converted.startswith("transformer."):
                        converted = "transformer." + converted
        
        return converted
    
    def test_lora_loading(self, model_path: str, lora_path: str) -> bool:
        """测试LoRA加载"""
        try:
            print(f"🔄 加载基础模型: {model_path}")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False
            )
            
            print(f"🔄 加载LoRA权重: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
            # 检查是否成功加载
            lora_found = False
            for name, module in pipe.transformer.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    print(f"✅ 检测到LoRA层: {name}")
                    lora_found = True
                elif 'lora' in name.lower():
                    print(f"🔍 可能的LoRA相关层: {name}")
            
            # 检查pipe的lora状态
            if hasattr(pipe, '_lora_scale') or hasattr(pipe, 'get_active_adapters'):
                try:
                    active_adapters = pipe.get_active_adapters() if hasattr(pipe, 'get_active_adapters') else []
                    print(f"🔍 活跃的适配器: {active_adapters}")
                    if active_adapters:
                        lora_found = True
                except:
                    pass
            
            return lora_found
            
        except Exception as e:
            print(f"❌ 加载测试失败: {str(e)}")
            return False
    
    def full_diagnostic_and_fix(self, lora_dir: str, model_path: str, output_dir: Optional[str] = None):
        """完整的诊断和修复流程"""
        print("=" * 60)
        print("🔍 开始LoRA诊断和修复")
        print("=" * 60)
        
        # 1. 诊断原始文件
        print("\n📋 步骤1: 诊断原始LoRA文件")
        diagnosis = self.diagnose_lora_files(lora_dir)
        
        print(f"找到文件: {len(diagnosis['files_found'])}")
        for file in diagnosis['files_found']:
            print(f"  - {file}")
        
        print(f"总张量数: {diagnosis['total_tensors']}")
        print(f"LoRA层数: {len(diagnosis['lora_layers'])}")
        
        if diagnosis['issues']:
            print("⚠️  发现问题:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
        
        # 2. 修复配置
        print("\n🔧 步骤2: 修复配置文件")
        self.fix_lora_adapter_config(lora_dir)
        
        # 3. 如果需要，转换格式
        if output_dir:
            print(f"\n🔄 步骤3: 转换权重格式到 {output_dir}")
            converted_path = self.convert_weights_format(lora_dir, output_dir)
            if converted_path:
                lora_dir = converted_path
        
        # 4. 测试加载
        print("\n🧪 步骤4: 测试LoRA加载")
        success = self.test_lora_loading(model_path, lora_dir)
        
        if success:
            print("🎉 LoRA加载成功!")
        else:
            print("❌ LoRA加载仍然失败")
            print("\n💡 建议:")
            print("1. 检查原始LoRA文件是否为SD3格式")
            print("2. 确认模型路径正确")
            print("3. 尝试使用不同的转换参数")
        
        return success

# 使用示例
if __name__ == "__main__":
    # 配置路径
    config = {
        'official_model_path': "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/",
        'lora_model_dir': "/home/lora_sd3_train_logs_0524_1600_0525_1600/lora_final/",
        'output_dir': "/home/lora_sd3_train_logs_0524_1600_0525_1600/lora_final_converted//"  # 可选：输出修复后的LoRA
    }
    
    # 创建诊断工具
    diagnostic = LoRADiagnosticTool()
    
    # 运行完整诊断和修复
    success = diagnostic.full_diagnostic_and_fix(
        lora_dir=config['lora_model_dir'],
        model_path=config['official_model_path'],
        output_dir=config['output_dir']
    )
    
    if success:
        print("\n✅ 现在可以正常使用LoRA了!")
    else:
        print("\n❌ 需要进一步检查原始LoRA文件")