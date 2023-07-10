import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

train_path = 'C:/Work/ASTAR/codes/CDA/CDA/Data/MNIST-M/MNIST-M_train.pt'
test_path = 'C:/Work/ASTAR/codes/CDA/CDA/Data/MNIST-M/MNIST-M_test.pt'

# Load the saved dataset
mnist_m_train = torch.load(train_path)
mnist_m_test = torch.load(test_path)

# Resize images and convert lists to tensors for better performance
mnist_m_train = [(interpolate(image.unsqueeze(0), size=(16, 16)).squeeze(0).view(1, -1), label) for image, label in mnist_m_train]
mnist_m_test = [(interpolate(image.unsqueeze(0), size=(16, 16)).squeeze(0).view(1, -1), label) for image, label in mnist_m_test]

# Use DataLoader to create batches for training and testing
batch_size = 32
trainloader = DataLoader(mnist_m_train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(mnist_m_test, batch_size=batch_size, shuffle=False)
