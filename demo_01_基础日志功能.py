import math
import random
import time
from torch.utils.tensorboard import SummaryWriter

# 1. 初始化 Writer，定义日志保存的目录
# 这里我们把日志存放在一个叫 "runs/my_first_experiment" 的文件夹下
writer = SummaryWriter(log_dir="runs/my_first_experiment")

print("开始模拟炼丹...")
total_epochs = 100

for epoch in range(total_epochs):
    # 模拟数据：Loss 呈指数衰减（加一点随机噪声模拟真实震荡）
    simulated_loss = 5.0 * math.exp(-0.05 * epoch) + random.uniform(-0.2, 0.2)
    
    # 模拟数据：Accuracy 逐渐逼近 100%（同样加点噪声）
    simulated_acc = 100.0 - 80.0 * math.exp(-0.08 * epoch) + random.uniform(-1.5, 1.5)

    # 2. 将数据写入 TensorBoard
    # 参数顺序：("图表名称", Y轴的具体数值, X轴的步数/Epoch)
    writer.add_scalar("Metrics/Loss", simulated_loss, epoch)
    writer.add_scalar("Metrics/Accuracy", simulated_acc, epoch)

    # 稍微暂停一下，模拟真实训练的耗时
    time.sleep(0.05) 

# 3. 养成好习惯，结束时关闭 writer
writer.close()
print("模拟结束！日志已生成。") 