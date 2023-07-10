import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data, resize it, and flatten it
transform = transforms.Compose([transforms.Resize((16,16)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Lambda(lambda x: torch.flatten(x)),
                                transforms.Lambda(lambda x: x.unsqueeze(0))])

# Download and load the full training and testing data
full_trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
full_testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

# Select a subset of the data
trainset = torch.utils.data.Subset(full_trainset, indices=range(0, 7500))
testset = torch.utils.data.Subset(full_testset, indices=range(0, 1500))

# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
