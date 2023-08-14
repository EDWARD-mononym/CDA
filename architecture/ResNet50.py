import torch
import torchvision.models as models

class ResNet50(torch.nn.Module):
    def __init__(self, configs):
        super(ResNet50, self).__init__()
        model = models.resnet50(weights=True)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Exclude the final FC layer.
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)  # Flattening before FC layer.
        return x