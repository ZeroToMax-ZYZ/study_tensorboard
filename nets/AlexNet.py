import torch
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AlexNet(nn.Module):
    '''
    note: not offical implementation of AlexNet.
    different part :
    - input size :    original (227*227*3) --> custom (224*224*3)
    - normalization : original (LRN:Local Response Normalization) --> custom (BN:Batch Normalization)
    - GPU groups :    original (2 groups) --> custom (1 group)
    '''
    def __init__(self, input_size=224, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            CBR(3, 96, 11, 4, 2), # [8, 96, 55, 55]
            nn.MaxPool2d(3, 2), # [8, 96, 27, 27]
            CBR(96, 256, 5, 1, 2), # [8, 256, 27, 27]
            nn.MaxPool2d(3, 2), # [8, 256, 13, 13]
            CBR(256, 384, 3, 1, 1), # [8, 384, 13, 13]
            CBR(384, 384, 3, 1, 1), # [8, 384, 13, 13]
            CBR(384, 256, 3, 1, 1), # [8, 256, 13, 13]
            nn.MaxPool2d(3, 2), # [8, 256, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    test_tensor = torch.randn(8, 3, 224, 224)
    model = AlexNet(input_size=224, num_classes=100)
    output = model(test_tensor)
    print(output.shape) # [8, 100]

