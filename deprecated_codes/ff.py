import torch
from diffusers import StableDiffusion3Pipeline
import os

def comprehensive_lora_verification(official_model_path, lora_model_dir):
    """全面验证LoRA权重加载"""
    print("🚀 开始加载官方模型...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        official_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        variant="fp16"
    ).to("cpu")
    print("✅ 官方模型加载完成")
    
    # 记录多个层的原始权重
    original_weights = {}
    layers_to_check = [
        ("transformer.transformer_blocks.0.attn.to_q", pipe.transformer.transformer_blocks[0].attn.to_q.weight),
        ("transformer.transformer_blocks.0.attn.to_k", pipe.transformer.transformer_blocks[0].attn.to_k.weight),
        ("transformer.transformer_blocks.0.attn.to_v", pipe.transformer.transformer_blocks[0].attn.to_v.weight),
        ("transformer.transformer_blocks.1.attn.to_q", pipe.transformer.transformer_blocks[1].attn.to_q.weight),
    ]
    
    for name, weight in layers_to_check:
        original_weights[name] = weight.data.clone()
        print(f"📋 记录原始权重 {name}: {weight.data[0, :3].tolist()}")
    
    print(f"\n🔄 加载LoRA权重: {lora_model_dir}")
    try:
        # 加载LoRA权重
        pipe.load_lora_weights(lora_model_dir)
        print("✅ LoRA权重文件加载成功")
        
        # 启用LoRA适配器（关键步骤！）
        if hasattr(pipe, 'enable_lora'):
            pipe.enable_lora()
            print("✅ LoRA适配器已启用")
        
        # 检查是否有LoRA适配器被加载
        if hasattr(pipe, 'get_active_adapters'):
            active_adapters = pipe.get_active_adapters()
            print(f"🔍 活跃的适配器: {active_adapters}")
        
    except Exception as e:
        print(f"❌ LoRA加载失败: {str(e)}")
        return False
    
    # 检查权重变化
    print(f"\n📊 检查权重变化...")
    any_change = False
    
    for name, original_weight in original_weights.items():
        current_weight = None
        try:
            # 重新获取当前权重
            if "transformer_blocks.0.attn.to_q" in name:
                current_weight = pipe.transformer.transformer_blocks[0].attn.to_q.weight.data
            elif "transformer_blocks.0.attn.to_k" in name:
                current_weight = pipe.transformer.transformer_blocks[0].attn.to_k.weight.data
            elif "transformer_blocks.0.attn.to_v" in name:
                current_weight = pipe.transformer.transformer_blocks[0].attn.to_v.weight.data
            elif "transformer_blocks.1.attn.to_q" in name:
                current_weight = pipe.transformer.transformer_blocks[1].attn.to_q.weight.data
                
            if current_weight is not None:
                weight_diff = torch.abs(original_weight - current_weight).mean().item()
                max_diff = torch.abs(original_weight - current_weight).max().item()
                
                print(f"📈 {name}:")
                print(f"   原始: {original_weight[0, :3].tolist()}")
                print(f"   当前: {current_weight[0, :3].tolist()}")
                print(f"   平均差异: {weight_diff:.8f}")
                print(f"   最大差异: {max_diff:.8f}")
                
                if weight_diff > 1e-6:  # 降低阈值
                    any_change = True
                    
        except Exception as e:
            print(f"❌ 检查层 {name} 时出错: {str(e)}")
    
    # 额外检查：验证LoRA参数
    print(f"\n🔍 检查LoRA参数状态...")
    lora_found = False
    
    # 检查transformer中的LoRA参数
    try:
        for name, module in pipe.transformer.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_found = True
                print(f"✅ 发现LoRA参数在: transformer.{name}")
                
                # 检查LoRA参数是否非零
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_a_norm = torch.norm(module.lora_A.weight).item()
                    lora_b_norm = torch.norm(module.lora_B.weight).item()
                    print(f"   LoRA_A权重范数: {lora_a_norm:.6f}")
                    print(f"   LoRA_B权重范数: {lora_b_norm:.6f}")
                    if lora_a_norm > 0 or lora_b_norm > 0:
                        any_change = True
                    break
    except Exception as e:
        print(f"⚠️ 检查transformer LoRA参数时出错: {str(e)}")
    
    # 也检查text_encoder中的LoRA参数
    try:
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            for name, module in pipe.text_encoder.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    lora_found = True
                    print(f"✅ 发现LoRA参数在: text_encoder.{name}")
                    break
    except Exception as e:
        print(f"⚠️ 检查text_encoder LoRA参数时出错: {str(e)}")
    
    # 检查LoRA适配器状态
    try:
        if hasattr(pipe, '_lora_loaded_modules'):
            loaded_modules = getattr(pipe, '_lora_loaded_modules', {})
            if loaded_modules:
                print(f"✅ 已加载的LoRA模块: {list(loaded_modules.keys())}")
                lora_found = True
    except Exception as e:
        print(f"⚠️ 检查LoRA适配器状态时出错: {str(e)}")
    
    if not lora_found:
        print("❌ 未找到LoRA参数，可能加载失败")
    
    return any_change or lora_found

def check_lora_files(lora_dir):
    """检查LoRA目录中的文件"""
    print(f"\n📁 检查LoRA目录: {lora_dir}")
    
    if not os.path.exists(lora_dir):
        print("❌ LoRA目录不存在")
        return False
        
    files = os.listdir(lora_dir)
    print(f"📋 目录中的文件: {files}")
    
    # 检查是否有预期的LoRA文件
    lora_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt', '.pth'))]
    if lora_files:
        print(f"✅ 找到LoRA权重文件: {lora_files}")
        return True
    else:
        print("❌ 未找到LoRA权重文件")
        return False

if __name__ == "__main__":
    config = {
        'official_model_path': "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/",
        'lora_model_dir': "/home/models/erikhsos-campusbier-sd3-lora/"
    }
    
    # 首先检查文件
    files_ok = check_lora_files(config['lora_model_dir'])
    
    if files_ok:
        # 运行验证
        print("\n" + "="*60)
        print("🔍 开始全面LoRA验证...")
        print("="*60)
        
        is_loaded = comprehensive_lora_verification(
            config['official_model_path'],
            config['lora_model_dir']
        )
        
        # 打印最终结果
        print("\n" + "="*60)
        print(f"🎯 最终验证结果: {'✅ 成功' if is_loaded else '❌ 失败'}")
        print("="*60)
        
        if not is_loaded:
            print("\n💡 可能的解决方案:")
            print("1. 检查LoRA文件格式是否正确")
            print("2. 尝试不同的LoRA加载方式")
            print("3. 确认LoRA是否与SD3兼容")
            print("4. 检查LoRA的缩放参数")
    else:
        print("❌ 无法继续验证，请检查LoRA文件路径和格式")