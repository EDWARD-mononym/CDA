import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os

class Custom_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std) #! mean = 0 & std = 1
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len

def create_dataloader(data_path, domain_id, configs, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Custom_Dataset(dataset_file, configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False    
        drop_last = False
    else:
        shuffle = configs.shuffle
        drop_last = configs.drop_last

    # Dataloaders
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=configs.train_params["batch_size"],
                             shuffle=shuffle, 
                             drop_last=drop_last, 
                             num_workers=0)

    return data_loader

data_path = "C:/Work/ASTAR/codes/CDA/CDA/Data/FD"

class Configs():
    def __init__(self) -> None:
        self.input_channels = 1
        self.normalize = True
        self.shuffle = True
        self.drop_last = True
        self.train_params = {"batch_size": 32}

config = Configs()

trainloader = create_dataloader(data_path, "CWRU_FE", config, dtype="train")
testloader = create_dataloader(data_path, "CWRU_FE", config, dtype="test")