import os
from numpy.random import f
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

################################################################
########################### UTILITY ############################
################################################################

data_dir = os.path.join(os.getcwd(), "dataset", "DomainNet")

transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def read_txt_file(file_path):
    samples, labels = [], []
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            parts = line.strip().split(' ') # Split the line by space
            file_path, class_label = parts[0], int(parts[1]) # Extract file path and class label
            samples.append(file_path)
            labels.append(class_label)
    return samples, labels

################################################################
########################### DATASET ############################
################################################################

class DomainNetDataset(Dataset):
    def __init__(self, domain, dtype):
        image_path = os.path.join(data_dir, f"{domain}_{dtype}")
        self.images, self.labels = read_txt_file(f"{image_path}.txt")
        self.transform = transforms_train if dtype == "train" else transforms_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = os.path.join(data_dir, self.images[idx])
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label

def create_dataloader(file_name, dtype):
    dataset = DomainNetDataset(file_name, dtype)
    dataloader = DataLoader(dataset,
                            batch_size=256,
                            shuffle=True
                            )
    return dataloader