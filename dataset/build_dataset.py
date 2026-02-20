import torch
from torch.utils.data import DataLoader, Subset

from dataset.augment import build_imagenet_transforms
from dataset.dataset_img100 import ImageNet100


def build_dataset(cfg):
    train_transform, val_transform = build_imagenet_transforms(cfg["input_size"])
    train_dataset = ImageNet100(cfg["train_path"], transform=train_transform)
    label_index = train_dataset._get_index()
    val_dataset = ImageNet100(cfg["val_path"], label_index, transform=val_transform)
    
    if cfg["debug_mode"] is not None:
        # fast debug mode, use a smaller subset
        test_size = int(len(train_dataset) * cfg["debug_mode"])
        indices = torch.randperm(len(train_dataset))[:test_size]
        train_dataset = Subset(train_dataset, indices)
        print("⚠️ debug mode : training dataset len: ", len(train_dataset))
        print("⚠️ debug mode : validation dataset len: ", len(val_dataset))
    else:
        print("training dataset len: ", len(train_dataset))
        print("validation dataset len: ", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["persistent_workers"], # 优化win, 复用进程
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["persistent_workers"], # 优化win, 复用进程
        prefetch_factor=2,
    )

    return train_loader, val_loader