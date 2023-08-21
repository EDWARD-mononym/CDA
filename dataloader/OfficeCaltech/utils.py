import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

office_labels = {'calculator': 0, 'headphones': 1, 'laptop_computer': 2,
                 'projector': 3, 'bike': 4, 'printer': 5, 'monitor': 6,
                 'mouse': 7, 'mug': 8, 'back_pack':9}

caltech_labels = {'027.calculator': 0, '101.head-phones': 1, '127.laptop-101': 2,
                  '238.video-projector': 3, '146.mountain-bike': 4, '161.photocopier': 5, '046.computer-monitor': 6,
                  '047.computer-mouse': 7, '041.coffee-mug': 8, '003.backpack': 9}

base_office_data_dir = os.path.join(os.getcwd(), "Data", "Office31")
caltech_data_dir = os.path.join(os.getcwd(), "Data", "Caltech256")

def load_office(domain):
    #* Create a list of filenames & class labels
    file_names = []
    labels = []
    office_domain__data_dir = os.path.join(base_office_data_dir, domain, "images")
    for object_name in office_labels:
        object_dir = os.path.join(office_domain__data_dir, object_name)

        with os.scandir(object_dir) as entries:
            temp_file_names = [str(object_dir + '/' + entry.name) for entry in entries if entry.is_file()]
            temp_labels = [office_labels[object_name] for _ in temp_file_names]
            file_names.extend(temp_file_names)
            labels.extend(temp_labels)

    #* Split data into training and testing sets
    test_size = 0.2  
    train_file_names, test_file_names, train_labels, test_labels = train_test_split(file_names, labels, test_size=test_size, random_state=42)

    #* Create the dataset
    train_dataset = CustomImageDataset(train_file_names, train_labels, transform=transform)
    test_dataset = CustomImageDataset(test_file_names, test_labels, transform=transform)

    #* Create the dataloader
    batch_size = 240
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

def load_caltech():
    #* Create a list of filenames & class labels
    file_names = []
    labels = []
    for object_name in caltech_labels:
        object_dir = os.path.join(caltech_data_dir, object_name)

        with os.scandir(object_dir) as entries:
            temp_file_names = [str(object_dir + '/' + entry.name) for entry in entries if entry.is_file()]
            temp_labels = [caltech_labels[object_name] for _ in temp_file_names]
            file_names.extend(temp_file_names)
            labels.extend(temp_labels)

    print(file_names)
    print(labels)

    #* Split data into training and testing sets
    test_size = 0.2  
    train_file_names, test_file_names, train_labels, test_labels = train_test_split(file_names, labels, test_size=test_size, random_state=42)

    #* Create the dataset
    train_dataset = CustomImageDataset(train_file_names, train_labels, transform=transform)
    test_dataset = CustomImageDataset(test_file_names, test_labels, transform=transform)

    #* Create the dataloader
    batch_size = 240
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


#* Creating custom dataset class & transformation
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

#* Defining the transfomation used
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])