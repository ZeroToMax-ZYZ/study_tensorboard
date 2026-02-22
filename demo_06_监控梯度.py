import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/histogram_experiment")
layer = nn.Linear(10, 10) # 模拟网络中的某一层

print("正在模拟权重更新过程...")
for epoch in range(10):
    # 模拟训练过程中权重的变化（加一些随机噪声代表梯度更新）
    layer.weight.data += torch.randn_like(layer.weight) * 0.1
    
    # 【核心代码】：记录这一层权重的分布
    # 第一个参数是名字，第二个参数是你想看的张量（通常是 model.layer.weight）
    writer.add_histogram("MyLinearLayer/Weights_Distribution", layer.weight, epoch)

writer.close()
print("直方图日志生成完毕！请去 TensorBoard 的 HISTOGRAMS 或 DISTRIBUTIONS 标签页查看。")