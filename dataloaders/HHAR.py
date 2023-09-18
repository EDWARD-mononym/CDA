import torch 

import os
import torch
from torch.utils.data import Dataset, DataLoader

dataset = os.path.join(os.getcwd(), "dataset/HHAR")

class CustomDataset(Dataset):
    def __init__(self, file_name, dtype):
        data_dict = torch.load(os.path.join(dataset, f"{dtype}_{str(file_name)}.pt"))

        self.x = data_dict['samples']
        self.y = data_dict['labels']

        # HHAR data is saved in (N_samples, Length, Channel[x, y, z]) format
        # To fit a 1D CNN, the data needs to be modified to the following format (N_samples, Channel[x, y, z], Length)
        self.x = self.x.transpose(0,2,1)

        # Convert to pytorch tensors
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long()

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

train_0 = create_dataloader(0, dtype="train")
test_0 = create_dataloader(0, dtype="test")

train_1 = create_dataloader(1, dtype="train")
test_1 = create_dataloader(1, dtype="test")

train_2 = create_dataloader(2, dtype="train")
test_2 = create_dataloader(2, dtype="test")

train_3 = create_dataloader(3, dtype="train")
test_3 = create_dataloader(3, dtype="test")

train_4 = create_dataloader(4, dtype="train")
test_4 = create_dataloader(4, dtype="test")

train_5 = create_dataloader(5, dtype="train")
test_5 = create_dataloader(5, dtype="test")

train_6 = create_dataloader(6, dtype="train")
test_6 = create_dataloader(6, dtype="test")

train_7 = create_dataloader(7, dtype="train")
test_7 = create_dataloader(7, dtype="test")

train_8 = create_dataloader(8, dtype="train")
test_8 = create_dataloader(8, dtype="test")