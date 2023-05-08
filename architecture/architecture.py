from torch import nn
from torch.autograd import Function

#############################################################################################################
############################################ FEATURE EXTRACTOR ##############################################
#############################################################################################################

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.CNN_params["input_channel"], configs.CNN_params["hidden_channel_1"], kernel_size=configs.CNN_params["kernel_1_size"],
                      stride=configs.CNN_params["stride_1"], bias=False, padding=(configs.CNN_params["kernel_1_size"] // 2)),
            nn.BatchNorm1d(configs.CNN_params["hidden_channel_1"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.CNN_params["hidden_channel_1"], configs.CNN_params["hidden_channel_2"], kernel_size=configs.CNN_params["kernel_2_size"], 
                      stride=configs.CNN_params["stride_2"], bias=False, padding=(configs.CNN_params["kernel_2_size"] // 2)),
            nn.BatchNorm1d(configs.CNN_params["hidden_channel_2"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.CNN_params["hidden_channel_2"], configs.CNN_params["output_channel"], configs.CNN_params["kernel_3_size"], 
                      stride=configs.CNN_params["stride_3"], bias=False, padding=(configs.CNN_params["kernel_3_size"] // 2)),
            nn.BatchNorm1d(configs.CNN_params["output_channel"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


#############################################################################################################
############################################### CLASSIFIER ##################################################
#############################################################################################################

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.CNN_params["output_channel"], configs.N_class)

    def forward(self, x):

        predictions = self.logits(x)

        return predictions

#############################################################################################################
############################################# DISCRIMINATOR #################################################
#############################################################################################################

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.CNN_params["output_channel"], configs.discriminator_param["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(configs.discriminator_param["hidden_dim"], configs.discriminator_param["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(configs.discriminator_param["hidden_dim"], 2)
        )

    def forward(self, input):
        out = self.layer(input)
        return out

#############################################################################################################
################################################# LOSSES ####################################################
#############################################################################################################


#############################################################################################################
################################################# DANN #####################################################
#############################################################################################################
#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None