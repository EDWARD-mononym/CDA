import torch
from collections import defaultdict

from algorithms.BaseModel import BaseModel

class DeepCoral(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        #* Losses
        self.coral_loss = CORAL()

    def update(self, dataloader, timestep): #* Update function updates the classifier and the feature extractor
        best_loss = float('inf')
        
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.hparams["N_epochs"]):
            losses = defaultdict(float)
            for (src_x, src_y), (trg_x, _) in zip(dataloader[0], dataloader[timestep]):
                src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

                #* Zero gradient
                self.feature_optimiser.zero_grad()
                self.classifier.zero_grad()

                #* Forward pass
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)
                trg_feat = self.feature_extractor(trg_x)

                #* Compute loss
                task_loss = self.task_loss(src_pred, src_y)
                coral_loss = self.coral_loss(src_feat, trg_feat)
                loss = self.configs.DeepCoral_params['coral_weight'] * coral_loss + self.configs.DeepCoral_params['task_weight'] * task_loss

                #* Backpropagation
                loss.backward()
                self.feature_optimiser.step()
                self.classifier_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(dataloader[0])
                losses["task_loss"] += task_loss.item() / len(dataloader[0])
                losses["coral_loss"] += coral_loss.item() / len(dataloader[0])

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), f"{self.configs.saved_models_path}/feature_extractor_{timestep}.pt")
                torch.save(self.classifier.state_dict(), f"{self.configs.saved_models_path}/classifier_{timestep}.pt")

        return epoch_losses



    #* Adapt function keeps the classifier constant
    #! Should only be called after training source model
    def adapt(self, dataloader, timestep):
        best_loss = float('inf')

        source_feature_extractor, _ = self.load_source_model() #* Loading source model
        
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.hparams["N_epochs"]):
            losses = defaultdict(float)
            for (src_x, _), (trg_x, _) in zip(dataloader[0], dataloader[timestep]):
                src_x, trg_x = src_x.to(self.device), trg_x.to(self.device)

                #* Zero gradient
                self.feature_optimiser.zero_grad()

                #* Forward pass
                src_feat = source_feature_extractor(src_x)
                trg_feat = self.feature_extractor(trg_x)

                #* Compute loss
                loss = self.coral_loss(src_feat, trg_feat)

                #* Backpropagation
                loss.backward()

                #* Update parameters
                self.feature_optimiser.step()

                #* Record the loss value
                losses["loss"] += loss.item() / len(dataloader[0])


            #* Learning rate scheduler
            self.feature_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), f"{self.configs.saved_models_path}/feature_extractor_{timestep}.pt")

        return epoch_losses

class CORAL(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss