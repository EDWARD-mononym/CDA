import torch
from torch.utils.data import ConcatDataset, DataLoader
from collections import defaultdict

from algorithms.BaseModel import BaseModel
from dataloader import Custom_Dataset, BATCH_SIZE

class IdealModel (BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

    def train(self, dataloaders):
        #* Combine dataset of all 
        dataset_list = [dl.dataset for dl in dataloaders]
        combined_dataset = ConcatDataset(dataset_list)
        modified_dataset = Custom_Dataset(combined_dataset)
        combined_dataloader = DataLoader(dataset=modified_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

        #* Training loop
        best_loss = float('inf')

        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.hparams["N_epochs"]):
            losses = defaultdict(float)
            for x, y in combined_dataloader:
                x, y = x.to(self.configs.device), y.to(self.configs.device)

                #* Zero the gradients
                self.feature_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Forward pass
                feature = self.feature_extractor(x)
                pred = self.classifier(feature)

                #* Compute loss
                loss = self.task_loss(pred, y)

                #* Backpropagation
                loss.backward()
                self.feature_optimiser.step()
                self.classifier_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(combined_dataloader)

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), f"{self.configs.saved_models_path}/feature_extractor_ideal.pt")
                torch.save(self.classifier.state_dict(), f"{self.configs.saved_models_path}/classifier_ideal.pt")

        return epoch_losses