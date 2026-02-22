import math
import random
from torch.utils.tensorboard import SummaryWriter

# 假设我们要对比三种不同的学习率
learning_rates = [0.1, 0.01, 0.001]

for lr in learning_rates:
    print(f"正在跑实验: 学习率 = {lr}")
    # 【核心技巧】：为每个参数组合，命名一个独立的子文件夹！
    writer = SummaryWriter(log_dir=f"runs/experiment_lr_{lr}")
    
    for epoch in range(50):
        # 模拟不同学习率带来的不同 Loss 下降速度
        # lr 越大，下降越快（这里只是简单的数学模拟）
        loss = 5.0 * math.exp(-lr * 10 * epoch) + random.uniform(-0.1, 0.1)
        
        writer.add_scalar("Train/Loss", loss, epoch)
    
    writer.close()

print("多组实验对比日志生成完毕！")