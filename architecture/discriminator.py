from torch import nn

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.CNN_params["feature_len"] * configs.CNN_params["output_channel"], configs.discriminator_param["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(configs.discriminator_param["hidden_dim"], configs.discriminator_param["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(configs.discriminator_param["hidden_dim"], 2)
        )

    def forward(self, input):
        out = self.layer(input)
        return out
