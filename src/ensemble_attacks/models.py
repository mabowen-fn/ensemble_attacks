import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16, densenet121, ResNet18_Weights, VGG16_Weights, DenseNet121_Weights
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_resnet18(num_classes=10, pretrained=True):
    """
    Get ResNet18 model.

    Args:
        num_classes: Number of output classes
        pretrained: If True, use ImageNet pretrained weights and fine-tune the final layer
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace final layer for target number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vgg16(num_classes=10, pretrained=True):
    """
    Get VGG16 model.

    Args:
        num_classes: Number of output classes
        pretrained: If True, use ImageNet pretrained weights and fine-tune the final layer
    """
    if pretrained:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Adapt for CIFAR (smaller images)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Replace classifier for target number of classes
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    else:
        model = vgg16(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    return model

def get_densenet121(num_classes=10, pretrained=True):
    """
    Get DenseNet121 model.

    Args:
        num_classes: Number of output classes
        pretrained: If True, use ImageNet pretrained weights and fine-tune the final layer
    """
    if pretrained:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_model(name: str, num_classes: int = 10, pretrained: bool = True, checkpoint_path: str = None):
    """
    Load a model by name.

    Args:
        name: Model name ("resnet18", "vgg16", or "densenet121")
        num_classes: Number of output classes
        pretrained: If True, use ImageNet pretrained weights (default: True)
        checkpoint_path: Optional path to a trained checkpoint file

    Returns:
        torch.nn.Module: The initialized model
    """
    name = name.lower()
    if name == "resnet18":
        model = get_resnet18(num_classes, pretrained)
    elif name == "vgg16":
        model = get_vgg16(num_classes, pretrained)
    elif name == "densenet121":
        model = get_densenet121(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using pretrained={pretrained} initialization")

    return model


