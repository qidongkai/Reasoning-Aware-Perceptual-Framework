import cv2
import numpy as np
import torch
import json
import os


def load_image_rgb(path):
    """加载RGB图像"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法加载图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(img, cfg):
    """图像预处理"""
    h, w = img.shape[:2]
    scale = cfg['image_size'] / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放
    img = cv2.resize(img, (new_w, new_h))

    # 填充为正方形
    canvas = np.zeros((cfg['image_size'], cfg['image_size'], 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = img

    # 归一化
    img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(cfg['mean']).view(3, 1, 1)
    std = torch.tensor(cfg['std']).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def draw_mask_overlay(img, mask):
    """绘制掩码叠加图"""
    color = np.zeros_like(img)
    color[mask > 0] = [0, 255, 0]  # 绿色掩码
    return cv2.addWeighted(img, 0.7, color, 0.3, 0)


def save_result_json(result, path):
    """保存结果为JSON"""
    # 处理numpy数组等不可序列化对象
    serializable_result = {}
    for k, v in result.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            serializable_result[k] = v.tolist()
        else:
            serializable_result[k] = v

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)