import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/pr_curve_experiment")

print("正在模拟 PR 曲线生成...")

# 模拟 1000 个测试样本
num_samples = 1000

# 真实的二分类标签 (0 或 1)
# 假设这是一个正样本较少的任务（比如 1000 个里只有 200 个正样本）
true_labels = torch.cat([torch.ones(200), torch.zeros(800)])

for epoch in range(5):
    # 模拟模型预测的概率 (0.0 到 1.0)
    # 随着 epoch 增加，模型预测越来越准：
    # 我们把真实标签和纯随机猜测混合，epoch 越大，预测越贴近真实标签
    random_guess = torch.rand(num_samples)
    
    # 准确率因子：随着 epoch 从 0 到 4 逐渐变大
    accuracy_factor = epoch / 4.0 
    
    # 生成预测概率并截断在 0~1 之间
    predicted_probs = true_labels.float() * accuracy_factor + random_guess * (1.0 - accuracy_factor)
    predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0)
    
    # 【核心代码】：记录 PR 曲线
    # 参数顺序：("图表名称", 真实的二值标签, 预测的概率值, 当前 Epoch)
    writer.add_pr_curve("Evaluations/PR_Curve", true_labels, predicted_probs, epoch)

writer.close()
print("PR 曲线日志生成完毕！请去 TensorBoard 的 PR CURVES 标签页查看。")