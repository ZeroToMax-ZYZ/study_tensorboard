import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None


    def forward(self, x):
        if self.shortcut is not None:
            identity = self.shortcut(x)
        else:
            identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # residual connection
        x = x + identity
        x = F.relu(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = ResBlock(64, 64)
        self.conv2_2 = ResBlock(64, 64)
        self.conv3_1 = ResBlock(64, 128, stride=2)
        self.conv3_2 = ResBlock(128, 128)
        self.conv4_1 = ResBlock(128, 256, stride=2)
        self.conv4_2 = ResBlock(256, 256)
        self.conv5_1 = ResBlock(256, 512, stride=2)
        self.conv5_2 = ResBlock(512, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


        
    def forward(self, x):
        # imagenet input size : 3*224*224
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x) # 64, 56, 56
        x = self.conv2_1(x) # 64, 56, 56
        x = self.conv2_2(x) # 64, 56, 56
        x = self.conv3_1(x) # 128, 28, 28
        x = self.conv3_2(x) # 128, 28, 28
        x = self.conv4_1(x) # 256, 14, 14
        x = self.conv4_2(x) # 256, 14, 14
        x = self.conv5_1(x) # 512, 7, 7
        x = self.conv5_2(x) # 512, 7, 7
        x = self.gap(x) # 512, 1, 1
        x = x.view(x.size(0), -1) # 512
        x = self.fc(x) # 100
        return x

if __name__ == '__main__':
    test_input = torch.randn(8, 3, 224, 224)
    model = ResNet18()
    output = model(test_input)
    print(output.shape)
    
        