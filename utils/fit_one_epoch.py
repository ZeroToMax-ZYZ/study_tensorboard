import torch
# from utils.logger import logger
from tqdm import tqdm
from utils.metric import cal_accuracy
from icecream import ic

import time

def fit_train_epoch(epoch, cfg, model, train_loader, loss_fn, optimizer):
    '''
    return: epoch_loss, epoch_top1(0-1), epoch_top5(0-1)
    '''
    model.train()

    train_loss = 0.0
    train_top1 = 0.0
    train_top5 = 0.0
    samples = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")

    for images, labels in train_bar:
        bs = images.shape[0]
        samples += bs

        img = images.to(cfg["device"])
        label = labels.to(cfg["device"])

        outputs = model(img)
        loss = loss_fn(outputs, label)
        top1, top5 = cal_accuracy(outputs, label)
        train_top1 += top1
        train_top5 += top5

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 恢复到整个bs的损失
        train_loss += loss.item() * bs

        # updata bar
        train_bar.set_postfix(loss=f"{train_loss/(samples):.4f}",
                              top1=f"{train_top1/samples*100:.2f}%")
        
    epoch_loss = train_loss / samples
    epoch_top1 = train_top1 / samples
    epoch_top5 = train_top5 / samples

    return epoch_loss, epoch_top1, epoch_top5

def fit_val_epoch(epoch, cfg, model, val_loader, loss_fn):
    '''
    return: epoch_loss, epoch_top1(0-1), epoch_top5(0-1)
    '''
    model.eval()

    val_loss = 0.0
    val_top1 = 0.0
    val_top5 = 0.0
    samples = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]")

    with torch.no_grad():
        for images, labels in val_bar:
            bs = images.shape[0]
            samples += bs

            img = images.to(cfg["device"])
            label = labels.to(cfg["device"])

            outputs = model(img)
            loss = loss_fn(outputs, label)
            top1, top5 = cal_accuracy(outputs, label)
            val_top1 += top1
            val_top5 += top5

            val_loss += loss.item() * bs

            # update bar
            val_bar.set_postfix(loss=f"{val_loss/(samples):.4f}",
                                top1=f"{val_top1/samples*100:.2f}%")

        epoch_loss = val_loss / samples
        epoch_top1 = val_top1 / samples
        epoch_top5 = val_top5 / samples

        return epoch_loss, epoch_top1, epoch_top5


def fit_one_epoch(epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler):
    '''
    return: train_loss, train_top1(0-1), train_top5(0-1),
            val_loss, val_top1(0-1), val_top5(0-1)
    '''
    start_time = time.time()
    train_loss, train_top1, train_top5 = fit_train_epoch(
        epoch, cfg, model, train_loader, loss_fn, optimizer
    )
    val_loss, val_top1, val_top5 = fit_val_epoch(
        epoch, cfg, model, val_loader, loss_fn
    )
    lr_scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time # (s)


    metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_top1": train_top1,
        "train_top5": train_top5,
        "val_loss": val_loss,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": epoch_time,
    }
    return metrics