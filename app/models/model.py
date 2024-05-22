import torch
from torch import nn
from torchvision.models import resnet50, resnet18

class SiameseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.cnn_model = resnet18(weights=None)
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(1, 1000)
        )

    def forward(self, x):
        out = self.cnn_model(x)
        out = out.view(out.size()[0], -1) # flatten to vector except for B channel
        return out