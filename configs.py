import torch

class Config:
    def __init__(self, dataset_config_class, hparam_class, algo_name):
        ###### DATASET CONFIGS ######
        self.sequence_len = dataset_config_class.sequence_len
        self.scenarios = dataset_config_class.scenarios
        self.class_names = dataset_config_class.class_names
        self.num_classes = dataset_config_class.num_classes
        self.shuffle = dataset_config_class.shuffle
        self.drop_last = dataset_config_class.drop_last
        self.normalize = dataset_config_class.normalize
        self.backbone = dataset_config_class.backbone

        #* model configs
        self.input_channels = dataset_config_class.input_channels
        self.kernel_size = dataset_config_class.kernel_size
        self.stride = dataset_config_class.stride
        self.dropout = dataset_config_class.dropout

        #* CNN and RESNET features
        self.mid_channels = dataset_config_class.mid_channels
        self.final_out_channels = dataset_config_class.final_out_channels
        self.features_len = dataset_config_class.features_len

        #* discriminator
        self.disc_hid_dim = dataset_config_class.disc_hid_dim
        self.DSKN_disc_hid = dataset_config_class.DSKN_disc_hid
        self.hidden_dim = dataset_config_class.hidden_dim

        #* MLP
        self.mlp_width = dataset_config_class.mlp_width
        self.mlp_depth = dataset_config_class.mlp_depth

        ###### HPARAM CONFIGS ######
        self.train_params = hparam_class.train_params
        self.alg_hparams = hparam_class.alg_hparams[algo_name]

        ###### EXPERIMENT CONFIGS ######
        self.num_runs = 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        