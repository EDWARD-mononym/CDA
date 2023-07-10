import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import bz2
import numpy as np
import os

class USPSDataset(Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform

        self.images = []
        self.labels = []

        with bz2.open(file, 'rt') as f:  # note that we open in text mode now
            for line in f:
                elements = line.strip().split()
                label = int(elements[0])
                image = np.zeros(256, dtype=np.float32)  # initialize a zero array with dtype float32
                for pixel in elements[1:]:
                    index, value = pixel.split(':')
                    image[int(index)-1] = float(value)  # subtract 1 from index because pixel indices are 1-based
                self.images.append(image)  # keep it as a 1D signal
                self.labels.append(label)

        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image[np.newaxis, :])  # add a new axis for num_channels

        return image, label
# Define a transform to normalize the data, resize it, and flatten it
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((16,16)), 
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Lambda(lambda x: torch.flatten(x)),
                                transforms.Lambda(lambda x: x.unsqueeze(0))])

datapath = os.path.join(os.getcwd(), "Data\\USPS")

trainset = USPSDataset(os.path.join(datapath, 'usps.bz2'), transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = USPSDataset(os.path.join(datapath, 'usps.t.bz2'), transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=True)
