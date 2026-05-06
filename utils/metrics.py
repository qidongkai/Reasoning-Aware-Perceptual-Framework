import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class Accuracy:
    """准确率指标"""

    def compute(self, preds, gts):
        return accuracy_score(gts, preds)


class F1Score:
    """F1分数指标"""

    def compute(self, preds, gts):
        return f1_score(gts, preds, average='macro')


class IoU:
    """IoU指标"""

    def compute(self, pred_mask, gt_mask):
        pred_mask = np.array(pred_mask).astype(bool)
        gt_mask = np.array(gt_mask).astype(bool)

        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        return inter / (union + 1e-8)


def calculate_f1(gts, preds):
    """计算F1分数（适配未知物种）"""
    return f1_score(gts, preds, average='binary')


def calculate_miou(masks_pred, masks_gt):
    """计算平均IoU"""
    if not masks_pred or not masks_gt:
        return 0.0

    ious = []
    for p, g in zip(masks_pred, masks_gt):
        iou = IoU().compute(p, g)
        ious.append(iou)

    return np.mean(ious)


def calculate_auroc(gts, preds):
    """计算AUROC"""
    try:
        return roc_auc_score(gts, preds)
    except:
        return 0.0