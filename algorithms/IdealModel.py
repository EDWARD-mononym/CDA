import torch
from torch.utils.data import DataLoader, Dataset # ConcatDataset
from collections import defaultdict
import os

from algorithms.BaseModel import BaseModel
from dataloader import Custom_Dataset, BATCH_SIZE

class IdealModel (BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

    def train(self, dataloaders):
        #* Combine dataset of all 
        dataset_list = [dataloaders[timestep].dataset for timestep in dataloaders]
        concat_dataset = ConcatDataset(dataset_list)
        combined_dataloader = DataLoader(dataset=concat_dataset,
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
                torch.save(self.feature_extractor.state_dict(), os.path.join(self.configs.saved_models_path, self.algo_name, f"feature_extractor_0.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(self.configs.saved_models_path, self.algo_name, f"classifier_0.pt"))

        return epoch_losses

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = torch.tensor(self.lengths).cumsum(dim=0).tolist()

    def __getitem__(self, index):
        index = int(index)  # Ensure that index is an integer
        for i, cum_len in enumerate(self.cumulative_lengths):
            if index < cum_len:
                if i > 0:
                    index -= self.cumulative_lengths[i - 1]
                return self.datasets[i][index]

    def __len__(self):
        return self.cumulative_lengths[-1]