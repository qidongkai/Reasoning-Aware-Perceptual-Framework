import torch
import os
import time
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Optional

from utils.logger import Logger
from utils.metrics import calculate_f1, calculate_miou, calculate_auroc
from utils.common import save_result_json

def save_model(model, optimizer, epoch, metrics, path):
    """保存模型"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }, path)

def save_config(config, path):
    """保存配置"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

class RAPFTrainer:
    """RAPF训练器"""
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 训练参数
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.warmup_epochs = config["warmup_epochs"]
        self.grad_clip = 5.0
        self.log_interval = 10
        self.save_dir = "checkpoints"
        self.log_dir = "logs"

        # 创建目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.warmup_epochs
        )

        # 日志和评估
        self.logger = Logger(self.log_dir)
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "val_f1": [],
            "val_acc": [],
            "val_miou": [],
            "val_auroc": []
        }

        # 保存配置
        save_config(config, os.path.join(self.log_dir, "config.json"))

    def train_epoch(self, epoch: int):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_cl = 0.0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        pbar.set_description(f"Train Epoch {epoch+1}/{self.epochs}")

        for step, batch in enumerate(pbar):
            # 加载数据
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            is_known = batch["is_known"].to(self.device)
            env_context = batch["environment"]

            # 前向传播
            outputs = self.model(
                images=images,
                labels=labels,
                env_context=env_context,
                is_known=is_known
            )

            # 计算损失
            loss = outputs["total_loss"]
            ce = outputs["cross_entropy_loss"]
            cl = outputs["contrastive_loss"]

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # 累计损失
            total_loss += loss.item()
            total_ce += ce.item()
            total_cl += cl.item()

            # 日志
            if step % self.log_interval == 0:
                self.logger.log_step(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    lr=self.optimizer.param_groups[0]["lr"]
                )

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce.item():.4f}",
                "cl": f"{cl.item():.4f}"
            })

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_ce = total_ce / len(self.train_loader)
        avg_cl = total_cl / len(self.train_loader)

        return {
            "avg_loss": avg_loss,
            "avg_ce": avg_ce,
            "avg_cl": avg_cl
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int):
        """验证单个epoch"""
        self.model.eval()
        preds, gts = [], []
        masks_pred, masks_gt = [], []

        for batch in tqdm(self.val_loader, desc="Validating"):
            img = batch['image'].to(self.device)
            label = batch['label'].item()
            is_known = batch['is_known'].item()
            env = batch['environment']

            # 推理
            res = self.model(img, env_context=[env])[0]

            # 收集结果
            preds.append(0 if res['label'] == 'unknown' else 1)
            gts.append(0 if not is_known else 1)

            # 掩码评估
            if 'mask' in res and batch.get('mask') is not None:
                masks_pred.append(res['mask'])
                masks_gt.append(batch['mask'].squeeze().cpu().numpy())

        # 计算指标
        metrics = {
            "accuracy": round(self.logger.acc.compute(gts, preds), 4),
            "f1_unknown": round(calculate_f1(gts, preds), 4),
            "miou": round(calculate_miou(masks_pred, masks_gt), 4),
            "auroc": round(calculate_auroc(gts, preds), 4)
        }

        self.logger.log_val(epoch, metrics)
        return metrics

    def save_best(self, metrics: Dict[str, float], epoch: int):
        """保存最优模型"""
        if metrics["f1_unknown"] > self.best_f1:
            self.best_f1 = metrics["f1_unknown"]
            self.best_epoch = epoch

            # 保存模型
            save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                path=os.path.join(self.save_dir, "RAPF_best.pth")
            )

            self.logger.log(f"Best model saved at epoch {epoch+1}, F1(unknown)={self.best_f1:.4f}")

    def update_scheduler(self, epoch: int):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

    def train(self):
        """主训练流程"""
        self.logger.log("Training Start")
        start_time = time.time()

        for epoch in range(self.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["avg_loss"])

            # 验证
            val_metrics = self.val_epoch(epoch)
            self.history["val_f1"].append(val_metrics["f1_unknown"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_miou"].append(val_metrics["miou"])
            self.history["val_auroc"].append(val_metrics["auroc"])

            # 保存最优模型
            self.save_best(val_metrics, epoch)

            # 更新学习率
            self.update_scheduler(epoch)

            # 日志
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics["avg_loss"],
                val_metrics=val_metrics
            )

        # 训练结束
        elapsed = time.time() - start_time
        self.logger.log(f"Training Finished. Best F1(unknown)={self.best_f1:.4f} at epoch {self.best_epoch+1}")
        self.logger.log(f"Total Time: {elapsed/60:.2f} min")

        # 保存训练历史
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)