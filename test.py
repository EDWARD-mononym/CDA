from torch.utils.data import DataLoader
import torch

# Custom Dataset Definition
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

# Transformation Function Definition
def transform(x):
    return x * 2  # Example transformation

# Create dataset and DataLoader
data = torch.Tensor([1, 2, 3, 4])
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=4)

# Apply transformation
for i, x in enumerate(dataloader):
    y = transform(x)
    dataloader.dataset[i] = y

# Check if the data in DataLoader has been transformed
transformed_data = [item.numpy()[0] for item in dataloader]
print(transformed_data)
