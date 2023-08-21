class OfficeCaltech(object): 
    def __init__(self):
        super(OfficeCaltech, self).__init__()
        self.sequence_len = 224*224
        self.scenarios = [("dslr", "amazon", "webcam", "caltech")]
        self.class_names = [str(i) for i in range(10)]
        self.num_classes = 10
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.backbone = "ResNet18"

        #* model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        #* CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 1
        self.features_len = 512

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