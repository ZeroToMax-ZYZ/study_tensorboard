# utf-8
from nets.AlexNet import AlexNet
from nets.yolov1 import YOLOv1_Classifier
from nets.yolov2 import YOLOv2_Classifier
from nets.resnet18 import ResNet18

def build_model(cfg):
    model_name = cfg["model_name"]
    print(" try to build model: ", model_name)
    if model_name == "AlexNet":
        model = AlexNet(input_size=cfg["input_size"], num_classes=100)

    elif model_name == "YOLOv1_backbone":
        model = YOLOv1_Classifier(num_classes=100, ic_debug=False)

    elif model_name == "YOLOv2_backbone":
        model = YOLOv2_Classifier(num_classes=100, ic_debug=False)

    elif model_name == "ResNet18":
        model = ResNet18(num_classes=100)
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model