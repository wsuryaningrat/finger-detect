import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class EmbeddingNet(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        n_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Sequential(

            nn.Linear(
                n_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            )

        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.embedding(x)

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x


class SiameseNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = EmbeddingNet()

    def forward(
        self,
        img1,
        img2
    ):

        emb1 = self.encoder(
            img1
        )

        emb2 = self.encoder(
            img2
        )

        return emb1, emb2