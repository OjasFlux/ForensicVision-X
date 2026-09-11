import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(
        self,
        spatial_features,
        frequency_features,
        output_classes=3
    ):
        super().__init__()

        total_features = (
            spatial_features
            + frequency_features
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                total_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                output_classes
            )
        )

    def forward(
        self,
        spatial,
        frequency
    ):

        combined = torch.cat(
            [spatial, frequency],
            dim=1
        )

        return self.classifier(
            combined
        )
