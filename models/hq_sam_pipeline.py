import torch
import numpy as np
import cv2
from segment_anything_hq import sam_model_registry, SamPredictor


class HQSAMPipeline:
    def __init__(self, config, device='cuda'):
        self.device = device
        self.nms_thresh = config['nms_threshold']
        self.conf_thresh = config['mask_conf_threshold']
        self.area_min = config['mask_area_min']
        self.area_max_ratio = config['mask_area_max_ratio']
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 初始化HQ-SAM
        sam = sam_model_registry[config['model_type']](checkpoint=config['sam_checkpoint'])
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def set_image(self, img):
        """设置输入图像"""
        self.predictor.set_image(img)

    def generate_masks(self, grid_size=16):
        """生成实例分割掩码"""
        if not hasattr(self.predictor, 'original_size'):
            return [], []

        h, w = self.predictor.original_size
        # 生成网格采样点
        coords = []
        for y in range(0, h, max(1, h // grid_size)):
            for x in range(0, w, max(1, w // grid_size)):
                coords.append([x, y])
        coords = np.array(coords)
        labels = np.ones(len(coords))

        # 预测掩码
        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True
        )

        # 选择每个点的最佳掩码
        best_masks, best_scores = [], []
        for i in range(len(masks)):
            best_idx = np.argmax(scores[i])
            best_masks.append(masks[i][best_idx])
            best_scores.append(scores[i][best_idx])

        # NMS过滤
        best_masks = np.array(best_masks)
        best_scores = np.array(best_scores)
        keep = self.nms(best_masks, best_scores)
        masks = best_masks[keep]
        scores = best_scores[keep]

        # 面积过滤和形态学处理
        masks = self.filter_by_area(masks, h, w)
        masks = self.morphology(masks)

        return masks, scores

    def nms(self, masks, scores):
        """掩码NMS"""
        areas = [m.sum() for m in masks]
        order = np.argsort(scores)[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            # 计算IoU
            iou = self.iou(masks[i], masks[order[1:]])
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def iou(self, m1, m2):
        """计算掩码IoU"""
        inter = (m1[None] * m2).sum(axis=(1, 2))
        union = m1.sum() + m2.sum(axis=(1, 2)) - inter
        return inter / (union + 1e-8)

    def filter_by_area(self, masks, h, w):
        """面积过滤"""
        max_area = h * w * self.area_max_ratio
        return [
            m for m in masks
            if self.area_min <= m.sum() <= max_area
        ]

    def morphology(self, masks):
        """形态学后处理"""
        res = []
        for m in masks:
            m = m.astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            res.append(m > 0)
        return res