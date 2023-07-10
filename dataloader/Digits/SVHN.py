import torch
from torch.utils.data import Dataset
from torchvision.datasets import SVHN
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

# Define a transform to convert the data to grayscale, resize it, and then normalize it.
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((16, 16)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ReshapeTransform((1, -1))])  # Reshape to (channels, signal_length)

# Download and load the full training and test datasets
full_trainset = SVHN(root='./data', split='train', download=True, transform=transform)
full_testset = SVHN(root='./data', split='test', download=True, transform=transform)

# Create subsets for the required number of samples
trainset = Subset(full_trainset, indices=range(0, 7500))
testset = Subset(full_testset, indices=range(0, 1500))

# Create data loaders for the subsets
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)
