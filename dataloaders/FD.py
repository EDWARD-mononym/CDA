import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # Load samples
        x_data = dataset["samples"]
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != 1:
            x_data = x_data.transpose(1, 2)

        # Normalize data
        data_mean = torch.mean(x_data, dim=(0, 2)).float()
        data_std = torch.std(x_data, dim=(0, 2)).float()
        normalise = Normalise(data_mean, data_std)
        self.transform = normalise.transform

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(1, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index

    def __len__(self):
        return self.len

class Normalise():
    def __init__(self, mean, std) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, x):
        return (x-self.mean) / self.std

def create_dataloader(data_path, domain_id, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Custom_Dataset(dataset_file)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False    
        drop_last = False
    else:
        shuffle = True
        drop_last = True

    # Dataloaders
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=256,
                             shuffle=shuffle, 
                             drop_last=drop_last, 
                             num_workers=0)

    return data_loader

#* Creating dataloaders for each domain
datasetpath = os.path.join(os.getcwd(), "dataset/FD")

CWRU_DE_train = create_dataloader(datasetpath, "CWRU_DE", dtype="train")
CWRU_DE_test = create_dataloader(datasetpath, "CWRU_DE",  dtype="test")

CWRU_FE_train = create_dataloader(datasetpath, "CWRU_FE", dtype="train")
CWRU_FE_test = create_dataloader(datasetpath, "CWRU_FE",  dtype="test")

PB_Artificial_train = create_dataloader(datasetpath, "PB_Artificial", dtype="train")
PB_Artificial_test = create_dataloader(datasetpath, "PB_Artificial",  dtype="test")

PB_Real_train = create_dataloader(datasetpath, "PB_Real", dtype="train")
PB_Real_test = create_dataloader(datasetpath, "PB_Real",  dtype="test")

MFPT_train = create_dataloader(datasetpath, "MFPT", dtype="train")
MFPT_test = create_dataloader(datasetpath, "MFPT", dtype="test")

# Conventional CWRU domains
zero_train = create_dataloader(datasetpath, "zero", dtype="train")
zero_test = create_dataloader(datasetpath, "zero",  dtype="test")

one_train = create_dataloader(datasetpath, "one", dtype="train")
one_test = create_dataloader(datasetpath, "one",  dtype="test")

two_train = create_dataloader(datasetpath, "two", dtype="train")
two_test = create_dataloader(datasetpath, "two",  dtype="test")

three_train = create_dataloader(datasetpath, "three", dtype="train")
three_test = create_dataloader(datasetpath, "three",  dtype="test")