import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#* Getting File paths
# Data folder & File list paths
folder_path = os.path.join(os.getcwd(), "Data/DomainNet")

train_path = os.path.join(folder_path, "painting_train.txt")
test_path = os.path.join(folder_path, "painting_test.txt")

# Read the file list into a DataFrame
train_df = pd.read_csv(train_path, header=None, delimiter=' ', names=['Path', 'Class'])
test_df = pd.read_csv(test_path, header=None, delimiter=' ', names=['Path', 'Class'])

# Create list of train test image & labels
train_img_list = train_df['Path'].tolist()
train_labels = train_df['Class'].tolist()

test_img_list = test_df['Path'].tolist()
test_labels = test_df['Class'].tolist()


#* Creating custom dataset class & transformation
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(folder_path, self.image_paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    # transforms.ColorJitter(),
    # transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#* Create the dataset
train_dataset = CustomImageDataset(train_img_list, train_labels, transform=transform)
test_dataset = CustomImageDataset(test_img_list, test_labels, transform=transform)

#* Create the dataloader
batch_size = 16
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
