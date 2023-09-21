import os
from pathlib import Path

from utils.get_loaders import get_loader

####### Info #######
#? This function loads the necesary dataloaders and create the save folders before calling the pretrain function for the algorithm
#? For detailed information on how the source model is trained, check the individual algorithm class in algorithms folder

def pretrain(algo_class, source_name, configs, device):
    train_loader = get_loader(configs["Dataset"]["Dataset_Name"], source_name, "train")
    test_loader = get_loader(configs["Dataset"]["Dataset_Name"], source_name, "test")

    save_folder = os.path.join(os.getcwd(), f"source_models/{configs['Dataset']['Dataset_Name']}/{configs['BackboneConfig']['Backbone']}")
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    algo_class.pretrain(train_loader, test_loader, source_name, save_folder, device)