import torch
import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义模型权重文件的完整本地路径
local_hf_model_dir = "/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32/" 


def setup_clip_model(model_path,device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CLIP模型权重文件未找到: {model_path}。请检查路径或文件名。")
    # 使用正确的CLIP模型和处理器
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    clip_model.eval()
    print("✅ CLIP模型加载成功")
    return clip_model,clip_processor

def setup_t5_model(sd3_model_path, device):
    """设置T5模型和视觉编码器"""
    print("🔄 加载SD3模型中的T5编码器...")
    t5_encoder = T5EncoderModel.from_pretrained(
        sd3_model_path,
        subfolder="text_encoder_3",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    t5_tokenizer = T5Tokenizer.from_pretrained(
            sd3_model_path, subfolder="tokenizer_3"
        )
        
    # 加载CLIP视觉编码器用于图像编码
    print("🔄 加载CLIP视觉编码器用于图像编码...")
    vision_model = CLIPVisionModel.from_pretrained(local_hf_model_dir).to(device)
    vision_processor = CLIPImageProcessor.from_pretrained(local_hf_model_dir)
    
    # 清理管道以节省内存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return t5_encoder, t5_tokenizer, vision_model, vision_processor



def load_images_and_prompts(image_folder, prompt_folder):
    """加载图像和对应的提示文本"""
    images, prompts, names = [], [], []
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for fname in sorted(image_files):
        base_name = os.path.splitext(fname)[0]
        prompt_path = os.path.join(prompt_folder, f"{base_name}.txt")
        
        if not os.path.exists(prompt_path):
            print(f"⚠️  提示文件未找到: {fname}, 跳过该文件")
            continue
            
        try:
            # 加载图像
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            
            # 加载对应 prompt
            with open(prompt_path, 'r', encoding='utf-8') as pf:
                prompt_text = pf.read().strip()
                prompts.append(prompt_text)
                
            names.append(fname)
            
        except Exception as e:
            print(f"⚠️  处理文件 {fname} 时出错: {e}")
            continue
    
    print(f"📊 成功加载 {len(images)} 个图像-文本对")
    return images, prompts, names

def compute_clip_score(image_folder, prompt_folder):
    clip_model,clip_processor = setup_clip_model(local_hf_model_dir,device)
    """计算CLIP相似度分数"""
    images, prompts, names = load_images_and_prompts(image_folder, prompt_folder)
    
    if len(images) == 0:
        raise RuntimeError("❌ 未找到有效的图像用于CLIP评分")
    
    similarities = []
    batch_size = 8  # 批处理以节省内存
    
    print("🔄 计算CLIP分数...")
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        # 使用CLIP处理器处理输入
        inputs = clip_processor(
            text=batch_prompts, 
            images=batch_images, 
            return_tensors="pt", 
            padding=True
        )
        
        # 将输入移动到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            
            # 获取对角线元素（每个图像与其对应文本的相似度）
            batch_similarities = torch.diag(logits_per_image)
            similarities.extend(batch_similarities.cpu().tolist())
    
    avg_similarity = sum(similarities) / len(similarities)
    print(f"✅ CLIP平均相似度: {avg_similarity:.4f}")
    
    return avg_similarity

def load_images_for_lpips(image_folder):
    """为LPIPS加载图像并转换为张量"""
    images = []
    transform = T.Compose([
        T.Resize((256, 256)),  # LPIPS通常使用256x256
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ])
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for fname in image_files:
        try:
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"⚠️  处理LPIPS图像 {fname} 时出错: {e}")
            continue
    
    return torch.stack(images) if images else torch.empty(0)

def compute_lpips(folder1, folder2):
    # 加载 LPIPS 模型
    lpips_model = lpips.LPIPS(net='alex').to(device)
    """计算两个文件夹之间图像的LPIPS距离"""
    print("🔄 加载图像用于LPIPS计算...")
    imgs1 = load_images_for_lpips(folder1)
    imgs2 = load_images_for_lpips(folder2)
    
    if len(imgs1) == 0 or len(imgs2) == 0:
        raise RuntimeError("❌ LPIPS计算时未找到有效图像")
    
    if len(imgs1) != len(imgs2):
        min_len = min(len(imgs1), len(imgs2))
        imgs1 = imgs1[:min_len]
        imgs2 = imgs2[:min_len]
        print(f"⚠️  图像数量不匹配，使用前 {min_len} 个图像")
    
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    
    distances = []
    print("🔄 计算LPIPS距离...")
    
    for i in tqdm(range(len(imgs1))):
        with torch.no_grad():
            d = lpips_model(imgs1[i:i+1], imgs2[i:i+1])
            distances.append(d.item())
    
    avg_distance = sum(distances) / len(distances)
    print(f"✅ LPIPS平均距离: {avg_distance:.4f}")
    
    return avg_distance

def compute_fid(folder1, folder2, batch_size=50):
    """计算两个文件夹之间的FID分数"""
    print("🔄 计算FID分数...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [folder1, folder2], 
            batch_size, 
            device, 
            2048
        )
        print(f"✅ FID分数: {fid_value:.2f}")
        return fid_value
    except Exception as e:
        print(f"❌ FID计算出错: {e}")
        raise


def compute_t5_score(image_folder, prompt_folder, sd3_model_path, device='cuda'):
    """计算T5相似度分数"""
    images, prompts, names = load_images_and_prompts(image_folder, prompt_folder)
    
    if len(images) == 0:
        raise RuntimeError("❌ 未找到有效的图像用于T5评分")
    
    # 设置模型
    t5_encoder, t5_tokenizer, vision_model, vision_processor = setup_t5_model(sd3_model_path, device)
    
    # 创建一个线性投影层来对齐嵌入维度
    # T5: 4096维 -> CLIP: 768维
    projection_layer = torch.nn.Linear(4096, 768).to(device)
    
    similarities = []
    batch_size = 4  # T5模型较大，使用较小的批次
    
    print("🔄 计算T5分数...")
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        text_inputs = t5_tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024  # T5可以处理更长的序列
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # 编码图像 - 使用CLIP视觉编码器
        image_inputs = vision_processor(batch_images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
            # 获取T5文本嵌入
            text_outputs = t5_encoder(**text_inputs)
            # 使用平均池化获得句子级别的表示
            text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
            
            # 获取CLIP图像嵌入
            image_outputs = vision_model(**image_inputs)
            image_embeddings = image_outputs.pooler_output
            
            # 将T5嵌入投影到CLIP嵌入空间
            text_embeddings_projected = projection_layer(text_embeddings.float())
            
            # 标准化嵌入
            text_embeddings_projected = F.normalize(text_embeddings_projected, p=2, dim=-1)
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            
            # 计算余弦相似度
            batch_similarities = torch.sum(text_embeddings_projected * image_embeddings, dim=-1)
            similarities.extend(batch_similarities.cpu().tolist())
    
    avg_similarity = sum(similarities) / len(similarities)
    print(f"✅ T5平均相似度: {avg_similarity:.4f}")
    
    return avg_similarity