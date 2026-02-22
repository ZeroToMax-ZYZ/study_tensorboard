import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/image_experiment")

print("正在生成图片日志...")

# 我们模拟 5 个 Epoch，每个 Epoch 记录一张图片
for epoch in range(5):
    # PyTorch 中图像张量的标准格式是：[Channels, Height, Width]
    # 这里我们生成一张 3通道 (RGB)、宽 100、高 100 的随机噪点图
    # torch.rand 会生成 0~1 之间的随机数，TensorBoard 能直接把 0~1 的张量渲染成彩色图片
    dummy_image = torch.rand(3, 100, 100)
    
    # 【核心代码】：记录单张图片
    writer.add_image("Debug_Images/Random_Noise", dummy_image, epoch)

writer.close()
print("图片日志生成完毕！")