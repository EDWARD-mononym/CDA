from torch import nn

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block_input = nn.Sequential(
            nn.Conv1d(configs["Dataset"]["input_channel"], configs["BackboneConfig"]["hidden_channels"], kernel_size=configs["BackboneConfig"]["kernel_size"],
                      stride=configs["BackboneConfig"]["stride"], bias=False, padding=(configs["BackboneConfig"]["kernel_size"] // 2)),
            nn.BatchNorm1d(configs["BackboneConfig"]["hidden_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs["BackboneConfig"]["dropout"])
        )

        self.conv_block_hidden = nn.Sequential(
            nn.Conv1d(configs["BackboneConfig"]["hidden_channels"], configs["BackboneConfig"]["hidden_channels"], kernel_size=configs["BackboneConfig"]["kernel_size"],
                      stride=configs["BackboneConfig"]["stride"], bias=False, padding=(configs["BackboneConfig"]["kernel_size"] // 2)),
            nn.BatchNorm1d(configs["BackboneConfig"]["hidden_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs["BackboneConfig"]["dropout"])
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs["BackboneConfig"]["feature_length"])

        self.configs = configs

    def forward(self, x_in):
        x = self.conv_block_input(x_in)
        for _ in range(self.configs["BackboneConfig"]["hidden_layers"]):
            x = self.conv_block_hidden(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat