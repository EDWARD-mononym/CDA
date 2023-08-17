import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the 2-layer MLP g(z)
class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(configs.features_len, configs.features_len)
        self.linear2 = nn.Linear(configs.features_len, configs.alg_hparams['len_encoded'])

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1)  # normalize so that ‖k(x)‖22 = 1
        return x