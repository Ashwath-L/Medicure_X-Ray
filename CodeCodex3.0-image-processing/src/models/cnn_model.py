import torch
import torch.nn as nn
from torchvision import models


class CustomChestCNN(nn.Module):
    """
    Lightweight baseline Convolutional Neural Network.
    Good for sanity checks and fast prototyping.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_densenet121(num_classes=2, pretrained=True, freeze_mode="most"):
    """
    Builds DenseNet-121 with selective layer freezing.
    freeze_mode options:
      - 'most' (recommended): Freezes features up to denseblock3/transition3.
                              Keeps denseblock4, norm5, and classifier trainable to avoid overfitting.
      - 'all': Freezes all convolutional feature extractors; trains only classifier head.
      - 'none': Trains all parameters end-to-end.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze_mode == "all":
        for param in model.features.parameters():
            param.requires_grad = False
    elif freeze_mode == "most":
        # Keep only denseblock4 and norm5 trainable; freeze earlier blocks
        trainable_feature_blocks = {"denseblock4", "norm5"}
        for name, child in model.features.named_children():
            if name not in trainable_feature_blocks:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


def build_model(model_name="densenet121", num_classes=2, pretrained=True, freeze_mode="most"):
    """Factory function for model creation with backbone freezing support."""
    model_name = model_name.lower()
    if model_name == "densenet121":
        return get_densenet121(num_classes=num_classes, pretrained=pretrained, freeze_mode=freeze_mode)
    elif model_name == "custom":
        return CustomChestCNN(num_classes=num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from ['densenet121', 'custom', 'resnet50']")
