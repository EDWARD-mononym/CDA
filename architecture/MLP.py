import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the 2-layer MLP g(z)
class MLP(nn.Module):
    def __init__(self, configs, in_dim=2048, hidden_dim=2048, out_dim=128):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1)  # normalize so that ‖k(x)‖22 = 1
        return x