import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

datasetpath = os.path.join(os.getcwd(), "dataset/UO")

class CustomDataset(Dataset):
    def __init__(self, file_name, dtype):
        data_dict = torch.load(os.path.join(datasetpath, f"{dtype}_{str(file_name)}.pt"))

        self.x = data_dict['samples']
        self.y = data_dict['labels']

        # Convert data type
        self.x = self.x.float()
        self.y = self.y.long()

        # Normalise the sample
        data_mean = torch.mean(self.x, dim=(0, 2), keepdim=True)
        data_std = torch.std(self.x, dim=(0, 2), keepdim=True)
        self.x = (self.x - data_mean) / data_std

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = self.x[idx]
        label = self.y[idx]
        return sample, label, idx
        
def create_dataloader(file_name, dtype):
    dataset = CustomDataset(file_name, dtype)
    dataloader = DataLoader(dataset,
                            batch_size=256,
                            shuffle=True
                            )
    return dataloader