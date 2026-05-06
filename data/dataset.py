import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class WildPlantOpenSet10K(Dataset):
    """WildPlantOpenSet-10K 数据集加载器"""
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root, split, 'images')
        self.anno_file = os.path.join(root, split, 'annotations.json')

        # 加载标注
        with open(self.anno_file, 'r', encoding='utf-8') as f:
            self.anno = json.load(f)

        self.samples = self.anno.get('samples', [])
        self.known_classes = self.anno.get('known_classes', [])
        self.unknown_classes = self.anno.get('unknown_classes', [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        fname = info['filename']
        img_path = os.path.join(self.img_dir, fname)

        # 加载图像
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 标注信息
        label = info.get('label', -1)
        is_known = info.get('is_known', False)
        family = info.get('family', '')
        genus = info.get('genus', '')
        env = info.get('environment', [])

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'is_known': torch.tensor(is_known, dtype=torch.bool),
            'family': family,
            'genus': genus,
            'environment': env,
            'filename': fname
        }