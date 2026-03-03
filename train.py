import torch

from nets.build_model import build_model
from dataset.build_dataset import build_dataset
from utils.optim_lr_factory import build_optimizer, build_lr_scheduler, build_loss_fn
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_logger, save_config
from utils.logger_tensorb import base_tensorb_logger, epoch_tensorb_logger, flatten_config

from torch.utils.tensorboard import SummaryWriter
from icecream import ic
import os
import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 获取当前device0的显卡型号
    GPU_model = torch.cuda.get_device_name(0)
    config = {
        "GPU_model": GPU_model,
        "exp_time": exp_time,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "08_YOLOv2_backbone_Tensorboard",
        "model_name": "YOLOv2_backbone",
        "save_interval": 10,
        # "train_path": r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train',
        # "val_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val",
        "train_path": r"/root/autodl-tmp/backbone_exp/datasets/Classification/ImageNet/train",
        "val_path": r"/root/autodl-tmp/backbone_exp/datasets/Classification/ImageNet/val",
        "model_path": None, # 加载训练好的权重,若为None则不加载
        # test model 
        "debug_mode": 0.1, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "input_size": 224,
        "batch_size": 128,
        "num_workers": 8,
        "persistent_workers": True, # 进程持久化,针对win平台
        "epochs": 5,
        "optimizer": {
            "type": "SGD",
            "lr": 0.1,
            "lr_scheduler": {
                "type": "StepLR",
                "step_size": 30,
                "gamma": 0.1,
            },
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        "loss_fn": "CrossEntropyLoss",
        # "optimizer": {
        #     "type": "Adam",
        #     "lr": 0.001,
        #     "lr_scheduler": {
        #         "type": "CosineAnnealingLR",
        #         "T_max": epochs,
        #         "eta_min": 1e-6,
        #     },
        #     "weight_decay": 1e-4,
        # }
    }

    config["exp_name"] += str("_" + exp_time)
    return config

def train():
    state = None
    cfg = base_config()
    save_config(cfg)
    # build tensorboard logger
    tb_path = os.path.join("logs", "logs_tensorboard", cfg["exp_name"])
    autudl_tb_path = os.path.join("/root/tf-logs", cfg["exp_name"])
    writer = SummaryWriter(log_dir=autudl_tb_path)

    model = build_model(cfg).to(cfg["device"])
    # 加载训练好的权重
    if cfg["model_path"] is not None:
        model.load_state_dict(torch.load(cfg["model_path"], map_location=cfg["device"]))
    
    train_loader, val_loader, train_dataset, val_dataset = build_dataset(cfg)

    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    loss_fn = build_loss_fn(cfg)
    # 调用tensorboard记录初始状态：数据增强之后的图像， 模型的结构，当前的训练参数
    base_tensorb_logger(writer, train_dataset, val_dataset, model, cfg)
    best_val_top1 = 0.0
    for epoch in range(cfg["epochs"]):
        metrics = fit_one_epoch(
            epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler
        )
        # 2. 更新最高 val_top1
        if metrics["val_top1"] > best_val_top1:
            best_val_top1 = metrics["val_top1"]
        # tensorboard logger
        epoch_tensorb_logger(writer, metrics, epoch)
        # save logs and model
        state = save_logger(model, metrics, cfg, state)
    # 3. 训练完全结束后，清洗 cfg 并记录 hparams
    flat_cfg = flatten_config(cfg)
    writer.add_hparams(
        hparam_dict=flat_cfg, 
        metric_dict={"hparam/best_val_top1": best_val_top1},
        run_name="."
    )
    # close tensorboard logger
    writer.close()

if __name__ == "__main__":
    if os.name == 'nt':  # 'nt'代表Windows系统
        torch.multiprocessing.set_start_method('spawn', force=True)
    train()
    

        
        







