
import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Model(nn.Module):

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_features=False
    ):
        super(MobileNetV2Model, self).__init__()

        if pretrained:
            weights = models.MobileNet_V2_Weights.DEFAULT
        else:
            weights = None

        self.model = models.mobilenet_v2(
            weights=weights
        )

        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)

