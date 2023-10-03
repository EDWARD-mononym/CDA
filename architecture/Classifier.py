from torch import nn

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(configs.output_channels * configs.feature_length, configs.num_class)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions