from typing import Tuple
import os
# 避免albumentations更新警告
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_imagenet_transforms(
    input_size=224,
    val_resize_short=None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[A.Compose, A.Compose]:
    """
    构建 ImageNet-1K 常用增强策略。
    """

    # 1) 按经典比例自动推导（强烈建议）
    if val_resize_short is None:
        val_resize_short = int(round(input_size * 256 / 224))
    # 2) 保险：val_resize_short 至少要 >= input_size，否则 CenterCrop 会越界
    val_resize_short = max(int(val_resize_short), input_size)

    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                height=input_size,
                width=input_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=1,
                p=1.0,
            ),

            # 2. 核心几何变换：水平翻转 (几乎零开销)
            A.HorizontalFlip(p=0.5),

            # 3. 颜色增强：模拟光照变化 (对 CPU 压力适中，但对精度很有用)
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.5,  # 稍微降低概率，不用每张都做
            ),
            
            # 4. (可选) 随机擦除：防止过拟合的利器 (计算量很小，只是填0)
            A.CoarseDropout(
                max_holes=1,
                max_height=int(0.25 * input_size),
                max_width=int(0.25 * input_size),
                min_holes=1,
                min_height=int(0.10 * input_size),
                min_width=int(0.10 * input_size),
                fill_value=0,
                p=0.2, 
            ),

            # 5. 归一化与转 Tensor
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # -------- val --------
    # 经典评测管线：短边缩放到 256 -> 中心裁剪 224
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(
                max_size=val_resize_short,
                interpolation=1,  # cv2.INTER_LINEAR
                p=1.0,
            ),
            # 兜底：即便某些异常图仍然导致尺寸不足，也不会再 crop 越界
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,   # cv2.BORDER_CONSTANT
                value=0,
                p=1.0,
            ),
            A.CenterCrop(height=input_size, width=input_size, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform

if __name__ == "__main__":
    train_transform, val_transform = build_imagenet_transforms()
