import torch
import torch.nn as nn


class FrequencyModel(nn.Module):
    """
    Placeholder for the FFT/DCT learning branch.
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):

        x = self.model(x)

        return torch.flatten(
            x,
            start_dim=1
        )
