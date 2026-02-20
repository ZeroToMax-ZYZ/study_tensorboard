import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np

from icecream import ic

class ImageNet100(Dataset):
    def __init__(self, root, label_index=None, transform=None):
        self.list_img, self.list_label, self.label_index = self.collate_path(root, label_index)
        self.transform = transform

    def _get_index(self):
        return self.label_index

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img_path = self.list_img[idx]
        label = self.list_label[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img_tensor = self.transform(image=img)["image"]
        else:
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


    def collate_path(self, paths, label_index=None):
        list_img = []
        list_label = []
        if label_index is None:
            label_index = [path for path in os.listdir(paths)]
        for path in os.listdir(paths):
            # path --> class name
            for img in os.listdir(os.path.join(paths, path)):
                # img --> image name
                img_path = os.path.join(paths, path, img)
                # check img exists
                if os.path.exists(img_path):
                    list_img.append(img_path)
                    list_label.append(label_index.index(path))
        return list_img, list_label, label_index



if __name__ == "__main__":

    train_path = r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train'
    val_path = r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val"
    
    train_dataset = ImageNet100(train_path)

    label_index = train_dataset._get_index()
    val_dataset = ImageNet100(val_path, label_index)
    
    ic(train_dataset[0].shape)

