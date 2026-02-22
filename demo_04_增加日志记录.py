import random
from torch.utils.tensorboard import SummaryWriter

print("开始模拟多组超参数搜索...")

# 假设我们做了 3 组不同的实验，每组配置不同
experiments = [
    {"lr": 0.1,   "batch_size": 32,  "optimizer": "SGD"},
    {"lr": 0.01,  "batch_size": 64,  "optimizer": "Adam"},
    {"lr": 0.001, "batch_size": 128, "optimizer": "Adam"}
]

for i, hparams in enumerate(experiments):
    print(f"正在记录实验 {i+1}，配置：{hparams}")
    
    # 依然是给每个实验一个独立的文件夹
    writer = SummaryWriter(log_dir=f"runs/hparams_search_exp_{i+1}")
    
    # 模拟训练结束后的最终结果
    # 我们假装第二组配置（Adam + lr=0.01）效果最好
    if hparams["optimizer"] == "Adam" and hparams["lr"] == 0.01:
        final_val_acc = 95.5
        final_val_loss = 0.12
    else:
        final_val_acc = random.uniform(70.0, 85.0)
        final_val_loss = random.uniform(0.8, 1.5)
        
    # 【核心代码】：将超参数字典与最终的指标字典绑定！
    # hparam_dict 传入你的参数配置；metric_dict 传入你关心的最终结果
    writer.add_hparams(
        hparam_dict=hparams, 
        metric_dict={
            "hparam/Final_Val_Accuracy": final_val_acc, 
            "hparam/Final_Val_Loss": final_val_loss
        }
    )
    
    writer.close()

print("超参数日志生成完毕！")