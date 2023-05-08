from torch import nn

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(configs.CNN_params["feature_len"] * configs.CNN_params["output_channel"], 
                                configs.dataset_params["N_class"])

    def forward(self, x):

        predictions = self.logits(x)

        return predictions