import torch
from torch import nn
import torchvision.models as models

class ResNet50(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()

    def forward(self, x_in):
        x = self.model(x_in)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat
