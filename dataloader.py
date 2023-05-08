import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os

from torch.utils.data.datapipes import datapipe

##### CONSTANTS #####
BATCH_SIZE = 32
#####################
class Custom_Dataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()

        #* Load samples and labels
        #* While target labels are not used in training, it is required in measuring performance
        x_data = dataset["samples"]
        x_data = x_data.unsqueeze(1) #? Adding input channel
        y_data = dataset["labels"]

        self.x_data = x_data.float()
        self.y_data = y_data.long()
        self.len = x_data.shape[0]


    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

    def __len__(self):
        return self.len

def load_scenarios(data_path, scenario):
    #? Scenario here will be in the form (S, T1, T2, T3, ...)
    train_loaders = {}
    test_loaders = {}

    for time_step, domain in enumerate(scenario):
        train_loaders[time_step], test_loaders[time_step] = create_dataloader(data_path, domain)

    return train_loaders, test_loaders

def create_dataloader(data_path, domain_name):
    train_files = torch.load(os.path.join(data_path, f"train_{domain_name}.pt"))
    train_set = Custom_Dataset(train_files)

    test_files = torch.load(os.path.join(data_path, f"test_{domain_name}.pt"))
    test_set = Custom_Dataset(test_files)

    train_loader = DataLoader(dataset=train_set,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_loader = DataLoader(dataset=test_set,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    return train_loader, test_loader