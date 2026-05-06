import torch
import torch.nn.functional as F


class IterationModule:
    """迭代优化模块"""

    def __init__(self, config, model):
        self.model = model
        self.lr = config['lr']
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=config.get('weight_decay', 1e-6)
        )

    def fine_tune(self, dataloader):
        """微调模型"""
        self.model.train()
        for batch in dataloader:
            img = batch['image'].to(self.model.device)
            label = batch['label'].to(self.model.device)

            # 前向传播
            feat = self.model.perception(img)
            logits = self.model.classifier(feat)
            loss = F.cross_entropy(logits, label)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()