from torch import nn

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.CNN_params["input_channel"], configs.CNN_params["hidden_channel_1"], kernel_size=configs.CNN_params["kernel_1_size"],
                      stride=configs.CNN_params["stride_1"], bias=False, padding=(configs.CNN_params["kernel_1_size"] // 2)),
            nn.BatchNorm1d(configs.CNN_params["hidden_channel_1"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.CNN_params["dropout"])
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

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.CNN_params["feature_len"])

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat
