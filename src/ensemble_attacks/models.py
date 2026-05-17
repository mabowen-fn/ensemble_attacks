import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16

def get_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vgg16(num_classes=10):
    """VGG16 model for CIFAR-10."""
    model = vgg16(weights=None)
    # Compute the flattened size after features
    # CIFAR-10: 32x32, after VGG features: 1x1x512
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, num_classes),
    )
    return model

