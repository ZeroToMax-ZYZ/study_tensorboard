import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid

import os 
import json



def denormalize(tensor, mean=None, std=None):
    """
    Denormalize a tensor using mean and std (适配 [B, C, H, W] 形状的张量).
    Args:
        tensor (Tensor): 待反归一化的张量，形状为 [B, C, H, W]
        mean (list): 各通道的均值，默认使用 ImageNet 均值
        std (list): 各通道的标准差，默认使用 ImageNet 标准差
    Returns:
        Tensor: 反归一化后的张量
    """
    # 设置默认的 ImageNet 均值和标准差
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    
    # 将 mean/std 转为张量，并调整形状为 [1, 3, 1, 1]，适配广播规则
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    # 反归一化核心逻辑：x = x * std + mean（与归一化 x = (x - mean)/std 互逆）
    denormalized_tensor = tensor.mul(std).add(mean)
    return denormalized_tensor




def base_tensorb_logger(writer, train_dataset, val_dataset,  model, cfg, train_img_count=5, val_img_count=5, epoch=0):
    # 保存模型结构
    img = train_dataset[0][0].to(device=cfg["device"])
    writer.add_graph(model, img.unsqueeze(0))
    # 保存训练集和验证集的图像数量
    writer.add_scalar('train_img_count', len(train_dataset), 0)
    writer.add_scalar('val_img_count', len(val_dataset), 0)
    
    # 保存数据增强之后的样例图像
    # 每9张图像以make_grid的拼为一张，然后保存train_img_count大张。注意要反归一化
    # 保存每9张图像的九宫格，并保存train_img_count个九宫格图像
    for count in range(train_img_count):
        train_images = []
        for i in range(9):  # 9宫格
            dataset_idx = count * 9 + i
            image, _ = train_dataset[dataset_idx]
            image = denormalize(image)  # 反归一化
            train_images.append(image)
        train_images = torch.cat(train_images)
        # 将图像列表转为 grid 形式，并保存到 TensorBoard
        grid_train = make_grid(train_images, nrow=3, padding=2)  # 3列，填充2
        writer.add_image(f'train_images', grid_train, global_step=count)

    for count in range(val_img_count):
        val_images = []
        for i in range(9):  # 9宫格
            dataset_idx = count * 9 + i
            image, _ = val_dataset[dataset_idx]
            image = denormalize(image)  # 反归一化
            val_images.append(image)

        # 将图像列表转为 grid 形式，并保存到 TensorBoard
        val_images = torch.cat(val_images)
        grid_val = make_grid(val_images, nrow=3, padding=2)  # 3列，填充2
        writer.add_image(f'val_images', grid_val, global_step=count)

    # 保存超参数 - 此时的保存只是单纯保存内容,不涉及到指标的绑定
    cfg_json_str = json.dumps(cfg, indent=4, ensure_ascii=False)
    writer.add_text('Config/Hyperparameters', f"```json\n{cfg_json_str}\n```", epoch)


def epoch_tensorb_logger(writer, metrics, epoch):
    # # 1. 将训练和验证的 Loss 画在同一张图里
    # writer.add_scalars('Metrics/Loss', {
    #     'train': metrics['train_loss'],
    #     'val': metrics['val_loss']
    # }, epoch)

    # # 2. 将训练和验证的 Top-1 准确率画在同一张图里
    # writer.add_scalars('Metrics/Accuracy_Top1', {
    #     'train': metrics['train_top1'],
    #     'val': metrics['val_top1']
    # }, epoch)

    # # 3. 将训练和验证的 Top-5 准确率画在同一张图里
    # writer.add_scalars('Metrics/Accuracy_Top5', {
    #     'train': metrics['train_top5'],
    #     'val': metrics['val_top5']
    # }, epoch)

    # # 4. 单独记录学习率，归类到 "Hyperparameters" 折叠面板下
    # writer.add_scalar('Hyperparameters/learning_rate', metrics['lr'], epoch)

    # # 5. 单独记录每个 epoch 的耗时，归类到 "System" 折叠面板下
    # writer.add_scalar('System/epoch_time', metrics['epoch_time'], epoch)
    # --- 方案一：标准分组记录法 ---
    # 注意斜杠前面的单词就是 TensorBoard 的大标题
    
    # Loss 分组
    writer.add_scalar('Loss/train', metrics['train_loss'], epoch)
    writer.add_scalar('Loss/val', metrics['val_loss'], epoch)

    # Top1 分组
    writer.add_scalar('Accuracy_Top1/train', metrics['train_top1'], epoch)
    writer.add_scalar('Accuracy_Top1/val', metrics['val_top1'], epoch)

    # Top5 分组
    writer.add_scalar('Accuracy_Top5/train', metrics['train_top5'], epoch)
    writer.add_scalar('Accuracy_Top5/val', metrics['val_top5'], epoch)

    # 学习率和时间
    writer.add_scalar('System/learning_rate', metrics['lr'], epoch)
    writer.add_scalar('System/epoch_time', metrics['epoch_time'], epoch)


def flatten_config(cfg):
    """
    将 cfg 字典中不支持的类型（如嵌套字典、None、列表等）转为字符串，
    防止 writer.add_hparams 报错。
    """
    flat_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            flat_cfg[k] = v
        else:
            flat_cfg[k] = str(v)  # 强转为字符串
    return flat_cfg