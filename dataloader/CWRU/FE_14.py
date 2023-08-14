import os

from dataloader.CWRU.utils import create_dataloader

data_path = os.path.join(os.getcwd(), "Data/CWRU")

trainloader = create_dataloader(data_path, "FE_14", dtype="train")
testloader = create_dataloader(data_path, "FE_14",  dtype="test")