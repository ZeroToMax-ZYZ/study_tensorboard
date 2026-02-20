import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    ResNet50/101/152 使用的 Bottleneck 残差块：
    主分支：1x1(降维) -> 3x3 -> 1x1(升维)
    shortcut：当 stride!=1 或 in_channels != out_channels 时，用 1x1 对齐
    expansion=4：最终输出通道 = base_channels * 4
    """
    expansion = 4

    def __init__(self, in_channels: int, base_channels: int, stride: int = 1):
        super(Bottleneck, self).__init__()

        out_channels = base_channels * self.expansion  # 升维后的最终通道

        # 1x1 conv：降维
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 3x3 conv：提取空间特征（这里 stride 可能是 2，用于下采样）
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)

        # 1x1 conv：升维
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # shortcut 分支（对齐形状）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # 这里不 ReLU，和 identity 相加后再 ReLU

        out = out + identity
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """
    ResNet50（Bottleneck 版本）默认配置：[3, 4, 6, 3]
    你现在希望把 conv4_x 做成可配置重复次数，从而一键得到 ResNet101（conv4_x=23）
    """

    def __init__(self, num_classes: int = 100, conv4_blocks: int = 6):
        super(ResNet50, self).__init__()

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x: base=64, out=256, 3 blocks
        self.conv2_1 = Bottleneck(64, 64, stride=1)     # 64 -> 256
        self.conv2_2 = Bottleneck(256, 64, stride=1)    # 256 -> 256
        self.conv2_3 = Bottleneck(256, 64, stride=1)    # 256 -> 256

        # conv3_x: base=128, out=512, 4 blocks（第一个 stride=2 下采样）
        self.conv3_1 = Bottleneck(256, 128, stride=2)   # 256 -> 512, 28x28
        self.conv3_2 = Bottleneck(512, 128, stride=1)   # 512 -> 512
        self.conv3_3 = Bottleneck(512, 128, stride=1)   # 512 -> 512
        self.conv3_4 = Bottleneck(512, 128, stride=1)   # 512 -> 512

        # =========================
        # conv4_x：改造为可配置重复次数（不使用 getattr）
        # base=256, out=1024
        # 第一个 block stride=2 下采样，其余 stride=1
        # =========================
        if conv4_blocks < 1:
            raise ValueError(f"conv4_blocks 必须 >= 1，当前为 {conv4_blocks}")

        self.conv4 = nn.ModuleList()
        self.conv4.append(Bottleneck(512, 256, stride=2))  # 第一个：512 -> 1024, 14x14
        for _ in range(conv4_blocks - 1):
            self.conv4.append(Bottleneck(1024, 256, stride=1))  # 后续：1024 -> 1024

        # conv5_x: base=512, out=2048, 3 blocks（第一个 stride=2 下采样）
        self.conv5_1 = Bottleneck(1024, 512, stride=2)  # 1024 -> 2048, 7x7
        self.conv5_2 = Bottleneck(2048, 512, stride=1)  # 2048 -> 2048
        self.conv5_3 = Bottleneck(2048, 512, stride=1)  # 2048 -> 2048

        # head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # imagenet input size : 3*224*224
        x = F.relu(self.bn1(self.conv1(x)))  # 64, 112, 112
        x = self.maxpool(x)                  # 64, 56, 56

        # conv2_x
        x = self.conv2_1(x)                  # 256, 56, 56
        x = self.conv2_2(x)                  # 256, 56, 56
        x = self.conv2_3(x)                  # 256, 56, 56

        # conv3_x
        x = self.conv3_1(x)                  # 512, 28, 28
        x = self.conv3_2(x)                  # 512, 28, 28
        x = self.conv3_3(x)                  # 512, 28, 28
        x = self.conv3_4(x)                  # 512, 28, 28

        # conv4_x（可变长度）
        # 第一个 block 后：1024, 14, 14；后续保持 1024, 14, 14
        for block in self.conv4:
            x = block(x)

        # conv5_x
        x = self.conv5_1(x)                  # 2048, 7, 7
        x = self.conv5_2(x)                  # 2048, 7, 7
        x = self.conv5_3(x)                  # 2048, 7, 7

        # GAP + FC
        x = self.gap(x)                      # 2048, 1, 1
        x = x.view(x.size(0), -1)            # 2048
        x = self.fc(x)                       # num_classes
        return x


class ResNet101(ResNet50):
    """
    ResNet101 只比 ResNet50 多了 conv4_x 的重复次数：
    ResNet50 : conv4_x = 6
    ResNet101: conv4_x = 23
    """
    def __init__(self, num_classes: int = 100):
        super(ResNet101, self).__init__(num_classes=num_classes, conv4_blocks=23)


if __name__ == "__main__":
    test_input = torch.randn(8, 3, 224, 224)

    model50 = ResNet50(num_classes=100, conv4_blocks=6)
    out50 = model50(test_input)
    print("ResNet50:", out50.shape)  # torch.Size([8, 100])

    model101 = ResNet101(num_classes=100)
    out101 = model101(test_input)
    print("ResNet101:", out101.shape)  # torch.Size([8, 100])
