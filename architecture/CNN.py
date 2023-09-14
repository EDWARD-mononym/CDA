from torch import nn

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs["Dataset"]["input_channel"], configs["BackboneConfig"]["hidden_channels"], kernel_size=configs["BackboneConfig"]["kernel_size"],
                      stride=configs["BackboneConfig"]["stride"], bias=False, padding=(configs["BackboneConfig"]["kernel_size"] // 2)),
            nn.BatchNorm1d(configs["BackboneConfig"]["hidden_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs["BackboneConfig"]["dropout"])
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs["BackboneConfig"]["hidden_channels"], configs["BackboneConfig"]["hidden_channels"]*1, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs["BackboneConfig"]["hidden_channels"]*1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs["BackboneConfig"]["hidden_channels"]*1, configs["BackboneConfig"]["output_channels"], kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs["BackboneConfig"]["output_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs["BackboneConfig"]["feature_length"])

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat