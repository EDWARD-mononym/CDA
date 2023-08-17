from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch

# Create the custom sampler
class CustomRandomSampler(RandomSampler):
    def __iter__(self):
        self.indices = list(super().__iter__())
        return iter(self.indices)

# Corrected CustomDatasetWithIndex class definition
class CustomDatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataloader, transform_func):
        self.data = dataloader.dataset
        self.transform_func = transform_func
        
        # Initialize tensor to store transformed values
        self.k = torch.zeros(len(self.data))
        
        # Iterate through the dataloader and apply the transformation
        for i, (x,) in enumerate(dataloader):
            idx = dataloader.sampler.indices[i] if isinstance(dataloader.sampler, CustomRandomSampler) else i
            self.k[idx] = transform_func(x.item())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, = self.data[idx]  # Corrected unpacking
        k = self.k[idx]

        return sample, k, idx  # Return both data and index

    def update_key(self, new_value, idx):
        self.k[idx] = new_value

# Create a simple dataset
data = torch.tensor([10, 20, 30, 40, 50])
dataset = TensorDataset(data)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
custom_sampler = CustomRandomSampler(dataloader.dataset)
dataloader_with_custom_sampler = DataLoader(dataloader.dataset, batch_size=1, sampler=custom_sampler)

transform_func = lambda x: x * 2  # Example transformation function

custom_dataset_with_index = CustomDatasetWithIndex(dataloader_with_custom_sampler, transform_func)

sample_0 = custom_dataset_with_index[0]
sample_1 = custom_dataset_with_index[1]
sample_2 = custom_dataset_with_index[2]

print(sample_0, sample_1, sample_2)