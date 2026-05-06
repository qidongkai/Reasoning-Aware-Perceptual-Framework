import torchvision.transforms as T

def get_train_transform(cfg):
    """训练集变换"""
    return T.Compose([
        T.Resize((cfg['image_size'], cfg['image_size'])),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(cfg['mean'], cfg['std'])
    ])

def get_val_transform(cfg):
    """验证集变换"""
    return T.Compose([
        T.Resize((cfg['image_size'], cfg['image_size'])),
        T.ToTensor(),
        T.Normalize(cfg['mean'], cfg['std'])
    ])

def get_test_transform(cfg):
    """测试集变换"""
    return T.Compose([
        T.Resize((cfg['image_size'], cfg['image_size'])),
        T.ToTensor(),
        T.Normalize(cfg['mean'], cfg['std'])
    ])