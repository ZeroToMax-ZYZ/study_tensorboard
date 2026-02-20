import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.augment import build_imagenet_transforms
from dataset.dataset_img100 import ImageNet100


# Keep the same default dataset path as train.py; edit below when using a different dataset.
TRAIN_PATH = Path(
    r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train"
)
SAMPLE_INDEX = None  # Set to an integer for a fixed sample
SEED = None  # Set to an integer for deterministic sampling
INPUT_SIZE = 224


def _select_sample(dataset: ImageNet100, sample_index: Optional[int], rng: random.Random) -> int:
    if sample_index is None:
        return rng.randrange(len(dataset))
    return sample_index % len(dataset)


def _read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _denormalize(image_tensor: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    # ToTensorV2 returns CHW, normalized; bring back to HWC in [0, 1]
    tensor = image_tensor.detach().cpu().numpy()
    tensor = tensor * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    tensor = np.clip(tensor, 0.0, 1.0)
    return np.transpose(tensor, (1, 2, 0))


def _gather_images(
    original: np.ndarray,
    augmented_tensors: List[torch.Tensor],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> List[np.ndarray]:
    images = [original.astype(np.uint8)]
    for tensor in augmented_tensors:
        img = (_denormalize(tensor, mean, std) * 255.0).astype(np.uint8)
        images.append(img)
    return images


def visualize(dataset_root: Path, sample_index: Optional[int], seed: Optional[int], input_size: int) -> None:
    rng = random.Random(seed)
    train_transform, _ = build_imagenet_transforms(input_size=input_size)

    dataset = ImageNet100(str(dataset_root))
    sample_idx = _select_sample(dataset, sample_index, rng)
    image_path = Path(dataset.list_img[sample_idx])
    label_idx = dataset.list_label[sample_idx]
    class_name = dataset.label_index[label_idx]

    original_image = _read_image(image_path)
    augmented_tensors = []
    for _ in range(9):
        augmented = train_transform(image=original_image)["image"]
        augmented_tensors.append(augmented)

    images = _gather_images(
        original=original_image,
        augmented_tensors=augmented_tensors,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    titles = [f"Original\n{class_name}"] + [f"Aug #{i}" for i in range(1, 10)]
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Dataset sample #{sample_idx} from {dataset_root}", fontsize=12)
    plt.tight_layout()
    plt.savefig(r"visual_aug.png", dpi=1000)



if __name__ == "__main__":
    visualize(
        dataset_root=TRAIN_PATH,
        sample_index=SAMPLE_INDEX,
        seed=SEED,
        input_size=INPUT_SIZE,
    )
