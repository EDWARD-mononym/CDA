class FD(object): 
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 1
        self.scenarios = [("MFPT", "CWRU_DE", "PB_Real", "PB_Artificial", "CWRU_FE"), 
                          ("CWRU_DE", "PB_Real", "PB_Artificial", "CWRU_FE", "MFPT"),
                          ("PB_Real", "PB_Artificial", "MFPT", "CWRU_DE", "CWRU_FE"),
                          ("PB_Real", "MFPT", "CWRU_DE", "CWRU_FE", "PB_Artificial")]
        self.class_names = ['Healthy', 'Inner', 'Outer']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.backbone = "CNN"

        #* model configs
        self.input_channels = 1
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        #* CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        #* TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        #* lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        #* discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        #* MLP
        self.mlp_width = 256
        self.mlp_depth = 4