# -*- coding: utf-8 -*-
"""
LeNet-5（更贴近论文实现）PyTorch 实现
==================================

修改点（按你的要求）：
- 不引入 BatchNorm
- 激活函数：ReLU -> tanh
- 池化：MaxPool2d -> AvgPool2d

结构：
Conv(5) -> AvgPool(2) -> Conv(5) -> AvgPool(2) -> Conv(5) -> FC(84) -> FC(num_classes)

默认输入：
- (N, 1, 32, 32) 适配 LeNet-5 常见输入（如 MNIST 做 padding 到 32x32）

注意：
- 论文原始 LeNet-5 的“子采样层 S2/S4”有可学习参数（缩放+偏置），AvgPool2d 是无参数的近似实现；
  如果你想严格复刻 S2/S4（带可学习系数），我也可以给你一个 Subsample 层实现。
"""

from typing import Tuple
import torch
import torch.nn as nn


class CT(nn.Module):
    """
    Conv + Tanh（不使用 BN，更贴近 LeNet 论文）
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.conv(x))


class LeNet(nn.Module):
    """
    LeNet-5 风格分类网络（论文风：tanh + avgpool，且无 BN）

    约定：
    - input_size 默认 32
    - in_channels 默认 1（灰度）
    - num_classes 默认 10
    """
    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.in_channels = in_channels

        # 32x32 输入下的经典尺寸变化：
        # 32 -> (Conv5) 28 -> (AvgPool2) 14
        # 14 -> (Conv5) 10 -> (AvgPool2) 5
        # 5  -> (Conv5) 1
        self.backbone = nn.Sequential(
            CT(in_channels, 6, 5, 1, 0),   # [N, 6, 28, 28]
            nn.AvgPool2d(2, 2),            # [N, 6, 14, 14]
            CT(6, 16, 5, 1, 0),            # [N, 16, 10, 10]
            nn.AvgPool2d(2, 2),            # [N, 16, 5, 5]
            CT(16, 120, 5, 1, 0),          # [N, 120, 1, 1]
        )

        # 论文：120 -> 84 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(120 * 1 * 1, 84),
            nn.Tanh(),
            nn.Linear(84, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def infer_backbone_output_shape(self, device: str = "cpu") -> Tuple[int, int, int]:
        """
        调试工具：推理 backbone 输出的 (C,H,W)，方便你改 input_size 时检查维度是否仍能到 1x1。
        """
        x = torch.randn(1, self.in_channels, self.input_size, self.input_size, device=device)
        y = self.backbone(x)
        return (y.shape[1], y.shape[2], y.shape[3])


if __name__ == "__main__":
    # LeNet 典型输入：灰度图 1x32x32
    test_tensor = torch.randn(8, 1, 32, 32)

    model = LeNet(input_size=32, in_channels=1, num_classes=10)
    output = model(test_tensor)

    print("output.shape =", output.shape)  # [8, 10]
    print("backbone out (C,H,W) =", model.infer_backbone_output_shape())
