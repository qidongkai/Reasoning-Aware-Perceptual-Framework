import torch
import yaml
import cv2
import numpy as np
import os
import json
import argparse
from models.rapf_model import RAPF
from data.dataset import WildPlantOpenSet10K
from data.transform import get_test_transform
from utils.common import (
    load_image_rgb,
    preprocess_image,
    draw_mask_overlay,
    save_result_json
)
from utils.metrics import calculate_f1, calculate_miou


def load_config(config_path: str = "configs/rapf_config.yaml"):
    """加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(config, device):
    """加载模型"""
    model = RAPF(config, device)
    weight_path = "checkpoints/RAPF_best.pth"

    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded weights from {weight_path}")
    else:
        print("Warning: No checkpoint found, using random initialization")

    model.eval()
    return model


def run_single_image(
        model,
        image_path: str,
        config,
        device,
        env_context=["mountain", "shrub", "grassland"]
):
    """单图像推理"""
    # 加载并预处理图像
    img_rgb = load_image_rgb(image_path)
    img_tensor = preprocess_image(img_rgb, config).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        results = model(
            images=img_tensor,
            env_context=[env_context]
        )
    result = results[0]

    # 可视化
    os.makedirs("outputs", exist_ok=True)
    if "mask" in result and result["mask"] is not None:
        vis = draw_mask_overlay(img_rgb, result["mask"])
        cv2.imwrite("outputs/result.jpg", vis[..., ::-1])

    # 保存结果
    save_result_json(result, "outputs/result.json")
    print("Result saved to outputs/")

    return result


def run_dataset_eval(model, config, device):
    """全数据集评估"""
    transform = get_test_transform(config)
    dataset = WildPlantOpenSet10K(
        root=config["dataset_root"],
        split="test",
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"]
    )

    preds, gts = [], []
    masks_pred, masks_gt = [], []

    print(f"Evaluating on test set (total {len(loader)} samples)...")
    for batch in loader:
        img = batch['image'].to(device)
        label = batch['label'].item()
        is_known = batch['is_known'].item()
        env = batch['environment']
        gt_mask = batch.get('mask')

        with torch.no_grad():
            res = model(img, env_context=[env])[0]

        # 收集预测结果
        preds.append(0 if res['label'] == 'unknown' else 1)
        gts.append(0 if not is_known else 1)

        # 收集掩码
        if 'mask' in res and gt_mask is not None:
            masks_pred.append(res['mask'])
            masks_gt.append(gt_mask.squeeze().numpy())

    # 计算指标
    f1 = calculate_f1(gts, preds)
    miou = calculate_miou(masks_pred, masks_gt)
    metrics = {
        'F1(unknown)': round(f1, 4),
        'mIoU': round(miou, 4),
        'Accuracy': round(sum([1 for p, g in zip(preds, gts) if p == g]) / len(gts), 4),
        'count': len(loader)
    }

    # 保存评估结果
    save_result_json(metrics, "outputs/eval.json")
    print("Evaluation results saved to outputs/eval.json")

    return metrics


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description="RAPF Test Script")
    parser.add_argument("--eval", action="store_true", help="Evaluate on full test set")
    parser.add_argument("--image", default="test.jpg", help="Path to test image")
    parser.add_argument("--env", nargs="+", default=["mountain", "shrub"], help="Environment context")
    args = parser.parse_args()

    # 加载配置和模型
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, device)
    print("Model loaded successfully")

    # 执行推理/评估
    if args.eval:
        metrics = run_dataset_eval(model, cfg, device)
        print("\nEvaluation Results:")
        print(json.dumps(metrics, indent=2))
    else:
        print(f"\nRunning inference on {args.image}...")
        res = run_single_image(model, args.image, cfg, device, args.env)
        print("\nInference Result:")
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()