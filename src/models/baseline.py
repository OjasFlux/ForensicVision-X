import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class ResNet50Baseline(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True
    ):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet50(
            weights=weights
        )

        input_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):

        return self.backbone(x)
