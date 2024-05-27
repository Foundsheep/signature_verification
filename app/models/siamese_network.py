import torch
from torch import nn
from torchvision.models import resnet18

class SiameseNetwork_OutputEmbedding(nn.Module):
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
    

class SiameseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn_model = resnet18(weights=None)
        
        # ------ sample tiny model
        # self.cnn_model = nn.Sequential(
        #     nn.Conv2d(3, 32, 3),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 64, 3),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 128, 3),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Linear(1, 1000)
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_model.fc.out_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward_once(self, x):
        out = self.cnn_model(x)
        out = out.view(out.size()[0], -1) # flatten to vector except for B channel
        return out

    def forward(self, input_1, input_2):
        out_1 = self.forward_once(input_1)
        out_2 = self.forward_once(input_2)

        # concatenate both images' fatures
        out = torch.cat((out_1, out_2), 1)

        # pass the features to linear layers
        out = self.fc(out)

        # get a scalar
        out = self.sigmoid(out).squeeze()
        return out