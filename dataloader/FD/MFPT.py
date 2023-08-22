import os

from dataloader.CWRU.utils import create_dataloader

data_path = os.path.join(os.getcwd(), "Data/FD")

trainloader = create_dataloader(data_path, "MFPT", dtype="train")
testloader = create_dataloader(data_path, "MFPT",  dtype="test")