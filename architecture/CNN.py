from torch import nn

class CNN(nn.Module):
    def __init__(self, configs, hyperparameters):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs["Dataset"]["input_channel"], hyperparameters["hidden_channels"], kernel_size=hyperparameters["kernel_size"],
                      stride=hyperparameters["stride"], bias=False, padding=(hyperparameters["kernel_size"] // 2)),
            nn.BatchNorm1d(hyperparameters["hidden_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(hyperparameters["dropout"])
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(hyperparameters["hidden_channels"], hyperparameters["hidden_channels"]*1, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(hyperparameters["hidden_channels"]*1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(hyperparameters["hidden_channels"]*1, hyperparameters["output_channels"], kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(hyperparameters["output_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(hyperparameters["feature_length"])

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat