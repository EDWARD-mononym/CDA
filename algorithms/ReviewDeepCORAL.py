from collections import defaultdict
from itertools import cycle
import os
import random
import torch
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo

class ReviewDeepCORAL(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.taskloss = torch.nn.CrossEntropyLoss()

        self.feature_extractor_optimiser = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.classifier_optimiser = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay
            )

        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser, 
                                      step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser, 
                                              step_size=configs.step_size, gamma=configs.gamma)

        self.memory = None

        self.hparams = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()
        self.classifier.train()

        epoch_memory_inputs, epoch_memory_labels = [], []

        if self.memory:
            combined_loader = zip(cycle(src_loader), trg_loader, cycle(self.memory))
        else:
            combined_loader = zip(cycle(src_loader), trg_loader)

        loss_dict = defaultdict(float)

        for step, (data) in enumerate(combined_loader):
            if self.memory:
                source, target, memory = data
                mem_x = memory[0].to(device)
            else:
                source, target = data
                memory, mem_x = None, None
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            #* Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            #* Forward pass
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            trg_feat = self.feature_extractor(trg_x)

            #* Compute loss
            classification_loss = torch.nn.functional.cross_entropy(src_pred, src_y)
            coral_loss = CORAL(src_feat, trg_feat)
            loss = classification_loss + self.hparams.coral_wt * coral_loss

            ###### ADDED REPLAY MEMORY #####
            if memory:
                #* Forward pass
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)
                trg_feat = self.feature_extractor(trg_x)

                #* Compute loss
                classification_loss = torch.nn.functional.cross_entropy(src_pred, src_y)
                coral_loss = CORAL(src_feat, trg_feat)
                mem_loss = classification_loss + self.hparams.coral_wt * coral_loss

                loss += mem_loss
            ####### END OF REPLAY MEMORY SECTION #####

            #* Step
            loss.backward()
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_classification_loss"] += classification_loss.item() / len(src_x)
            loss_dict["avg_coral_loss"] += coral_loss.item() / len(src_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        if epoch == self.configs.n_epoch-1:
            # Select a portion of the current data for the memory
            indices = list(range(len(epoch_memory_inputs)))
            random.shuffle(indices)
            # n_to_store = int(self.configs.alpha * len(epoch_memory_inputs))
            n_to_store = int(0.05 * len(epoch_memory_inputs))
            selected_indices = indices[:n_to_store]

            selected_inputs = [epoch_memory_inputs[i] for i in selected_indices]
            selected_labels = [epoch_memory_inputs[i] for i in selected_indices]
            
            selected_inputs = torch.cat(selected_inputs)
            selected_labels = torch.cat(selected_labels)

            new_memory = DataLoader(TensorDataset(selected_inputs, selected_labels), batch_size=trg_loader.batch_size, shuffle=True)

            # Update memory
            if self.memory is None:
                self.memory = new_memory
            else:
                concatenated_dataset = ConcatDataset([self.memory.dataset, new_memory.dataset])
                self.memory = DataLoader(concatenated_dataset, batch_size=trg_loader.batch_size)

        return loss_dict

        return loss_dict

    def pretrain(self, train_loader, test_loader, source_name, save_path, device, evaluator):
        best_acc = -1.0
        print(f"Training source model")
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}/{self.n_epoch}')

            self.feature_extractor.to(device)
            self.classifier.to(device)
            self.feature_extractor.train()
            self.classifier.train()
            running_loss = 0
            for step, data in enumerate(train_loader):
                x, y = data[0], data[1]
                x, y = x.to(device), y.to(device)

                # Zero grads
                self.feature_extractor_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                # Forward pass
                pred = self.classifier(self.feature_extractor(x))

                # Loss
                loss = self.taskloss(pred, y)
                loss.backward()

                # Step
                self.feature_extractor_optimiser.step()
                self.classifier_optimiser.step()

                running_loss += loss.item()

            # Adjust learning rate
            self.fe_lr_scheduler.step()
            self.classifier_lr_scheduler.step()

            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
            print("-" * 30)  # Print a separator for clarity

            #* Save best model
            epoch_acc = evaluator.test_domain(self, test_loader)
            if epoch_acc > best_acc:
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss