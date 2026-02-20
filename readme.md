
# Project: ImageClassification

> [!NOTE] 核心说明
> 本项目是一个基于 **PyTorch** 实现的通用图像分类框架，旨在提供标准化的训练与推理流程。
> * **核心特性**：集成 `Albumentations` 增强管道、支持经典骨干网络（Backbone）、解耦数据加载与模型定义。
> * **适用场景**：ImageNet-100 子集实验、自定义数据集分类任务。
> 
> 

---

#  1. Project Structure | 项目结构

```bash
ImageClassification
├── dataset/                # 数据集根目录
│   └── train/              # 训练集（按类别分文件夹）
├── nets/                   # 网络模型定义 (ResNet, VGG, AlexNet, etc.)
├── utils/                  # 工具类 (Logger, Metrics, Checkpoint)
├── train.py                # 训练主入口
├── inference.py            # 推理/测试脚本
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明文档

```

# 2. how to install | 安装说明
**安装命令：**

```bash
# 1. 创建虚拟环境 (推荐)
conda create -n img_cls python=3.8.10
conda activate img_cls

# 2. 安装依赖
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

```

---

##  3. Dataset Preparation | 数据准备

本项目采用标准的 `ImageFolder` 格式。请确保数据集目录结构严格遵循以下规范：

```text
dataset/
├── train/
│   ├── n01440764/          # 类别名/ID
│   │   ├── n01440764_1.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ...
└── val/                    # (可选) 验证集结构同上

```

> [!WARNING] 注意事项
> * 图片扩展名支持 `.jpg`, `.jpeg`, `.png`。

---

##  4. Usage | 使用说明

tobecontinue...

### 4.2 Inference (推理)

tobecontinue...


##  5. Supported Models | 支持模型

- [x] AlexNet-BN
- [ ] GoogLeNet-BN
- [x] YOLOv1_backbone
- [ ] VGG-BN
- [ ] ResNet-34


##  6. Features & Roadmap | 特性与规划
- [x] **Data Augmentation**: 集成 `Albumentations` 库，支持强数据增强（Cutout, Mixup 等）。
- [x] **augmentation visualization**: 可视化数据增强效果。
- [x] **Checkpoint**: 自动保存最优模型（Best Model）与最后轮次模型（Last Model）。
- [ ] **随机种子**: 固定随机种子，保证实验可复现。
- [ ] **Logging**: 支持 TensorBoard 实时监控 Loss 与 Accuracy 曲线。
- [ ] **AMP**: 混合精度训练支持（待开发）。
- [ ] **DDP**: 分布式多卡训练支持（待开发）。

