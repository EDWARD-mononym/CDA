class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        self.sequence_len = 128
        self.scenarios = [(8, 7, 2, 1, 3),
                          (0, 2, 1, 3, 4),
                          (0, 8, 7, 6, 5),
                          (5, 1, 2, 7, 6)]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        #* model configs
        self.input_channels = 3
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
