from torch import nn

class Classifier(nn.Module):
    def __init__(self, configs, hyperparameters):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(hyperparameters["output_channels"] * hyperparameters["feature_length"], configs["Dataset"]["num_class"])

    def forward(self, x):

        predictions = self.logits(x)

        return predictions