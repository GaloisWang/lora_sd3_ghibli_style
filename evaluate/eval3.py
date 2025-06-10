import os
from utils import compute_t5_score, compute_lpips, compute_fid

def main():
    # 配置路径
    base_folder = "/home/sd3_lora_compare/output_official/"
    lora_folder = "/home/sd3_lora_compare/output_finetuned_fulldata_060216e5/"
    prompts_dir = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli/captions_gstyle/"
    real_style_folder = "/root/autodl-tmp/HuggingFace_Datasets/ItzLoghotXD--Ghibli-filtered-deduplicate/"
    sd3_model_path = "/root/autodl-tmp/models/stabilityai--stable-diffusion-3-medium-diffusers/"
    
    # 检查路径是否存在
    paths_to_check = {
        "基础模型输出": base_folder,
        "LoRA模型输出": lora_folder,
        "提示文件": prompts_dir,
        "真实风格图像": real_style_folder
    }
    
    print("🔍 检查路径...")
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"❌ {name}路径不存在: {path}")
            return
        else:
            print(f"✅ {name}: {path}")
    
    print("\n" + "="*60)
    
    try:
        # 1. T5 Score评估
        print("📊 1. T5 Score 评估")
        print("-" * 30)
        t5_base = compute_t5_score(base_folder, prompts_dir,sd3_model_path)
        t5_lora = compute_t5_score(lora_folder, prompts_dir,sd3_model_path)
        
        print(f"🔹 基础模型T5分数: {t5_base:.4f}")
        print(f"🔹 LoRA模型T5分数: {t5_lora:.4f}")
        
        if t5_lora > t5_base:
            print(f"✅ LoRA模型在文本-图像对齐方面表现更好 (+{t5_lora-t5_base:.4f})")
        else:
            print(f"⚠️  基础模型在文本-图像对齐方面表现更好 (+{t5_base-t5_lora:.4f})")
        
        print("\n" + "="*60)
        
        # 2. LPIPS评估
        print("📊 2. LPIPS 感知差异评估")
        print("-" * 30)
        lpips_val = compute_lpips(base_folder, lora_folder)
        print(f"🔹 LPIPS分数: {lpips_val:.4f}")
        
        if lpips_val < 0.3:
            print("✅ 两个模型生成的图像感知上相似")
        elif lpips_val < 0.6:
            print("⚠️  两个模型生成的图像有中等程度的感知差异")
        else:
            print("❗ 两个模型生成的图像感知差异较大")
        
        print("\n" + "="*60)
        
        # 3. FID评估
        print("📊 3. FID 图像质量评估")
        print("-" * 30)
        fid_base = compute_fid(base_folder, real_style_folder)
        fid_lora = compute_fid(lora_folder, real_style_folder)
        
        print(f"🔹 基础模型FID: {fid_base:.2f}")
        print(f"🔹 LoRA模型FID: {fid_lora:.2f}")
        
        if fid_lora < fid_base:
            print(f"✅ LoRA模型更接近真实风格 (FID降低 {fid_base-fid_lora:.2f})")
        else:
            print(f"⚠️  基础模型更接近真实风格 (FID降低 {fid_lora-fid_base:.2f})")
        
        print("\n" + "="*60)
        
        # 总结
        print("📋 评估总结")
        print("-" * 30)
        print(f"T5分数对比: 基础({t5_base:.4f}) vs LoRA({t5_lora:.4f})")
        print(f"LPIPS差异: {lpips_val:.4f}")
        print(f"FID分数对比: 基础({fid_base:.2f}) vs LoRA({fid_lora:.2f})")
        
        # 综合评估
        t5_better = "LoRA" if t5_lora > t5_base else "基础"
        fid_better = "LoRA" if fid_lora < fid_base else "基础"
        
        print(f"\n🎯 结论:")
        print(f"   • 文本对齐: {t5_better}模型更好")
        print(f"   • 风格相似: {fid_better}模型更好")
        print(f"   • 感知差异: {lpips_val:.4f} (越小越相似)")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()