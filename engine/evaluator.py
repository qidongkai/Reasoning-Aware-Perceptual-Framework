import torch
from tqdm import tqdm
from utils.metrics import Accuracy, F1Score, IoU, calculate_miou, calculate_auroc

class Evaluator:
    """评估器"""
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.acc = Accuracy()
        self.f1 = F1Score()
        self.iou = IoU()

    def evaluate_all(self):
        """全量评估"""
        self.model.eval()
        preds, gts = [], []
        masks_pred, masks_gt = [], []

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Evaluating"):
                img = batch['image'].to(self.device)
                label = batch['label'].item()
                is_known = batch['is_known'].item()
                env = batch['environment']
                gt_mask = batch.get('mask')

                # 推理
                res = self.model(img, env_context=[env])[0]

                # 收集预测和标注
                preds.append(0 if res['label'] == 'unknown' else 1)
                gts.append(0 if not is_known else 1)

                # 掩码评估
                if 'mask' in res and gt_mask is not None:
                    masks_pred.append(res['mask'])
                    masks_gt.append(gt_mask.squeeze().cpu().numpy())

        # 计算指标
        acc = self.acc.compute(preds, gts)
        f1 = self.f1.compute(preds, gts)
        miou = calculate_miou(masks_pred, masks_gt)
        auroc = calculate_auroc(gts, preds)

        return {
            "accuracy": round(acc, 4),
            "f1_unknown": round(f1, 4),
            "miou": round(miou, 4),
            "auroc": round(auroc, 4)
        }