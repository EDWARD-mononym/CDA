import torch
from easydict import EasyDict

CNN_params = {"input_channel": 1, 
              "hidden_channel_1": 64, "kernel_1_size": 5, "stride_1": 1,
              "hidden_channel_2": 128, "kernel_2_size": 8, "stride_2": 1,
              "output_channel": 128, "kernel_3_size": 8, "stride_3": 1,
              "dropout": 0.5, "feature_len": 1}

DISCRIMINATOR_params = {"hidden_dim": 64}

DATASET_params = {"N_class": 4}

HYPER_params = {"learning_rate": 0.0005, "weight_decay": 1e-4, "step_size": 50, "lr_decay": 0.5, "N_epochs": 40}

# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")

###### Algorithm specific parameters #######
DEEPCORAL_params = {"task_weight": 0.4386, "coral_weight": 5.936}
DANN_params = {"task_weight": 0.9603, "domain_weight": 0.9238}
### END OF Algorithm specific parameters ###

CONFIG_DICT = {"CNN_params": CNN_params,
               "discriminator_param": DISCRIMINATOR_params,
               "dataset_params": DATASET_params,
               "hparams": HYPER_params,
               "device": DEVICE,
               "saved_models_path": "C:/Work/ASTAR/codes/CDA/CDA/Trained_models",
               "DeepCoral_params": DEEPCORAL_params,
               "DANN_params": DANN_params,
               "datapath": "Data/CWRU",
               "scenario": ("CWRU_7", "CWRU_14", "CWRU_21")}

CONFIG = EasyDict(CONFIG_DICT)