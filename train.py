import torch
import yaml
from torch.utils.data import DataLoader
from data.dataset import WildPlantOpenSet10K
from data.transform import get_train_transform, get_val_transform
from models.rapf_model import RAPF
from engine.trainer import RAPFTrainer

def load_config(config_path="configs/rapf_config.yaml"):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 加载配置
    cfg = load_config()
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])

    # 加载数据集
    train_tf = get_train_transform(cfg)
    val_tf = get_val_transform(cfg)

    train_dset = WildPlantOpenSet10K(
        root=cfg['dataset_root'],
        split='train',
        transform=train_tf
    )
    val_dset = WildPlantOpenSet10K(
        root=cfg['dataset_root'],
        split='val',
        transform=val_tf
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    # 初始化模型
    model = RAPF(cfg, device)

    # 初始化训练器
    trainer = RAPFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )

    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()