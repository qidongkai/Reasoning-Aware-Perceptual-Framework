import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from timm.models import create_model
from models.hq_sam_pipeline import HQSAMPipeline


class CLIPDINOv2Fusion(nn.Module):
    """CLIP + DINOv2 特征融合模块"""

    def __init__(self, config):
        super().__init__()
        # 加载预训练模型
        self.clip = CLIPVisionModel.from_pretrained(f"openai/{config['clip_model']}")
        self.dinov2 = create_model(config['dinov2_model'], pretrained=True)
        self.alpha = config['fusion_weight']

        # 冻结预训练权重
        for p in self.clip.parameters():
            p.requires_grad_(False)
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """前向传播：融合特征"""
        f_clip = self.clip(x).last_hidden_state[:, 0, :]  # CLIP cls特征
        f_dino = self.dinov2(x)  # DINOv2 特征
        return self.alpha * f_clip + (1 - self.alpha) * f_dino


class PerceptionModule(nn.Module):
    """感知模块：特征提取 + 掩码生成"""

    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self.extractor = CLIPDINOv2Fusion(config).to(device)
        self.mask_gen = HQSAMPipeline(config, device)
        self.sim_thresh = config['similarity_threshold']

    def forward(self, img):
        """提取全局特征"""
        return self.extractor(img)

    def generate_masks(self, img_np):
        """生成实例分割掩码"""
        self.mask_gen.set_image(img_np)
        return self.mask_gen.generate_masks()

    def extract_mask_feature(self, img, mask):
        """提取掩码区域的特征"""
        # 掩码加权特征提取
        B, C, H, W = img.shape
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0)
        mask = nn.functional.interpolate(mask, (H, W), mode='nearest')

        # 特征提取
        feat = self.extractor(img)
        mask_feat = (img * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
        return (feat + mask_feat) / 2  # 融合全局和局部特征

    def classify_open_set(self, feat, text_embeds):
        """开集分类：判断是否为已知物种"""
        sim = torch.cosine_similarity(feat, text_embeds)
        max_sim, idx = torch.max(sim, dim=0)
        return max_sim.item() > self.sim_thresh, max_sim.item(), idx.item()