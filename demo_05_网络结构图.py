import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 随便写一个简单的 CNN 玩玩
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

writer = SummaryWriter(log_dir="runs/graph_experiment")
model = SimpleCNN()

# 【核心代码】：提供一个虚拟输入，TensorBoard 需要靠跑一遍前向传播来追踪计算图
dummy_input = torch.randn(1, 3, 32, 32)
writer.add_graph(model, dummy_input)

writer.close()
print("网络结构图已生成！请去 TensorBoard 的 GRAPHS 标签页查看。")