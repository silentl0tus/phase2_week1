import torch
import torch.nn as nn
from torchvision.models import resnet50

def MyResNet(num_classes=30):
    model = resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    return model

from torchvision.models import ResNet50_Weights
def get_preprocess():
    return ResNet50_Weights.DEFAULT.transforms()