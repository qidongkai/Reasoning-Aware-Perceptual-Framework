import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

from models.perception import PerceptionModule
from models.retrieval import RetrievalModule
from models.reasoning import DSReasoner
from models.decision import DecisionEngine
from models.iteration import IterationModule


class RAPF(nn.Module):
    """RAPF主模型：Reasoning-Aware Perceptual Framework"""

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.feat_dim = config['feat_dim']
        self.num_known = config['num_known_species']
        self.temp = config['temperature']
        self.cl_loss_w = config['contrastive_loss_weight']
        self.ce_loss_w = 1.0 - self.cl_loss_w

        # 初始化子模块
        self.perception = PerceptionModule(config, device)
        self.retrieval = RetrievalModule(config)
        self.reasoner = DSReasoner(config)
        self.decision = DecisionEngine(config)
        self.iteration = IterationModule(config, self)

        # 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.LayerNorm(self.feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feat_dim // 2, self.num_known)
        ).to(device)

        # 知识嵌入（文本/原型）
        self.register_buffer(
            'text_embeds',
            torch.randn(self.num_known, self.feat_dim, device=device)
        )
        self.register_buffer(
            'proto_embeds',
            torch.randn(self.num_known, self.feat_dim, device=device)
        )

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.cl_loss = nn.CrossEntropyLoss()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_knowledge(self, text_embeds, proto_embeds):
        """更新知识嵌入"""
        self.text_embeds = text_embeds.to(self.device)
        self.proto_embeds = proto_embeds.to(self.device)

    def train_step(self, images, labels, is_known, env_context):
        """训练步骤"""
        B = images.shape[0]
        total_ce = 0.0
        total_cl = 0.0

        for i in range(B):
            img = images[i:i + 1]
            label = labels[i]
            known = is_known[i]
            env = env_context[i]

            # 生成掩码并提取特征
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * self.config['std'] + self.config['mean']) * 255
            img_np = img_np.astype(np.uint8)

            masks, _ = self.perception.generate_masks(img_np)
            if len(masks) == 0:
                continue

            feat = self.perception.extract_mask_feature(img, masks[0])

            # 已知物种：计算损失
            if known:
                # 交叉熵损失
                logits = self.classifier(feat)
                ce = self.ce_loss(logits, label.unsqueeze(0))
                total_ce += ce

                # 对比损失
                pos = self.proto_embeds[label].unsqueeze(0)
                neg_idx = torch.randint(0, self.num_known, (16,), device=self.device)
                neg = self.proto_embeds[neg_idx]
                queue = torch.cat([pos, neg], dim=0)

                sim_mat = F.cosine_similarity(feat, queue, dim=-1) / self.temp
                cl = self.cl_loss(sim_mat.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=self.device))
                total_cl += cl

        # 平均损失
        avg_ce = total_ce / B if B > 0 else 0.0
        avg_cl = total_cl / B if B > 0 else 0.0
        total_loss = self.ce_loss_w * avg_ce + self.cl_loss_w * avg_cl

        return {
            "total_loss": total_loss,
            "cross_entropy_loss": avg_ce,
            "contrastive_loss": avg_cl
        }

    @torch.no_grad()
    def infer_step(self, images, env_context):
        """推理步骤"""
        B = images.shape[0]
        results = []

        for i in range(B):
            img = images[i:i + 1]
            env = env_context[i] if isinstance(env_context[0], list) else env_context

            # 1. 感知：特征提取 + 掩码生成
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * self.config['std'] + self.config['mean']) * 255
            img_np = img_np.astype(np.uint8)

            masks, scores = self.perception.generate_masks(img_np)
            if len(masks) == 0:
                results.append({
                    "label": "unknown",
                    "confidence": 1.0,
                    "mask": None
                })
                continue

            # 提取特征
            feat = self.perception.extract_mask_feature(img, masks[0])
            mask_score = float(np.mean(scores)) if scores else 0.0

            # 2. 开集分类
            is_known, vis_sim, idx = self.perception.classify_open_set(
                feat, self.text_embeds
            )

            # 3. 检索：Top-K相似物种
            candidates = self.retrieval(feat, env)
            cand_scores = [c.get('similarity', 0.0) for c in candidates]

            # 4. 推理：D-S证据融合
            mass, conflict, label = self.reasoner.infer(
                visual=[vis_sim],
                knowledge=cand_scores[:3],
                env=[self.retrieval.env_sim(c.get('environment', []), env) for c in candidates[:3]]
            )

            # 5. 决策：生成最终结果
            confidence = float(np.max(mass)) if mass is not None else 0.0
            result = self.decision.generate(
                label=label,
                confidence=confidence,
                conflict=conflict,
                is_known=is_known,
                similarity=vis_sim,
                candidates=candidates,
                mask_score=mask_score
            )
            result['mask'] = masks[0]  # 添加掩码
            results.append(result)

        return results

    def forward(
            self,
            images: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            env_context: Optional[List] = None,
            is_known: Optional[torch.Tensor] = None
    ):
        """前向传播"""
        if self.training and labels is not None and is_known is not None:
            return self.train_step(images, labels, is_known, env_context)
        else:
            return self.infer_step(images, env_context or [])