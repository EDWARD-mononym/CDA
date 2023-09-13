from torch import nn

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(configs["ClassifierConfig"]["input_size"], configs["Dataset"]["num_class"])

    def forward(self, x):

        predictions = self.logits(x)

        return predictions