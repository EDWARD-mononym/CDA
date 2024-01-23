from collections import defaultdict
import itertools
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo

class CUA(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # optimizer and scheduler
        self.feature_extractor_optimiser = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        # optimizer and scheduler
        self.classifier_optimiser = torch.optim.Adam(
            self.classifier.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser, step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser, step_size=configs.step_size, gamma=configs.gamma)

        # hparams
        self.hparams = configs

        # Alignment losses
        self.taskloss = torch.nn.CrossEntropyLoss()

        # Added memory
        self.memory = None

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()
        self.classifier.train()

        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        loss_dict = defaultdict(float)

        if self.memory:
            combined_loader = zip(itertools.cycle(src_loader), trg_loader, itertools.cycle(self.memory))
        else:
            combined_loader = zip(itertools.cycle(src_loader), trg_loader)

        for step, (data) in enumerate(combined_loader):
            if self.memory:
                source, target, memory = data
                mem_x = memory[0].to(device)
                mem_y = memory[1].to(device)
            else:
                source, target = data
                memory, mem_x = None, None

            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            #* Check if all are same size
            if memory:
                if mem_x.shape[0] != trg_x.shape[0]: # This happens towards the end where the remaining data in target is less than batchsize
                    continue

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
            loss = classification_loss + self.hparams.coral_wt * coral_loss + self.hparams.mem_wt * mem_loss

            if memory:
                # extract memory features
                mem_feat = self.feature_extractor(mem_x)
                mem_pred = self.classifier(mem_feat)

                # calculate memory entropy loss
                mem_cls_loss = self.taskloss(mem_pred, mem_y)

                mem_loss = self.hparams.mem_wt * mem_cls_loss
                loss = loss + mem_loss


            loss.backward()
            #* Step
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_classification_loss"] += classification_loss.item() / len(src_x)
            loss_dict["avg_coral_loss"] += coral_loss.item() / len(src_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()s

        # Save target to memory
        if epoch == self.configs.n_epoch-1:
            subsampled_dataset = random_sample(trg_loader, self.feature_extractor, self.classifier, device, n = 0.01)
            repeated_sampler = RepeatSampler(subsampled_dataset)
            new_memory = DataLoader(subsampled_dataset, batch_size=trg_loader.batch_size, sampler=repeated_sampler, drop_last=False)

            # Update memory
            if self.memory is None:
                self.memory = new_memory
            else:
                concatenated_dataset = ConcatDataset([self.memory.dataset, new_memory.dataset])
                repeated_sampler = RepeatSampler(concatenated_dataset)
                self.memory = DataLoader(concatenated_dataset, batch_size=trg_loader.batch_size, sampler=repeated_sampler, drop_last=False)

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
            acc_dict = evaluator.test_all_domain(self)
            epoch_acc = acc_dict[source_name]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

            #* Log epoch acc
            evaluator.update_epoch_acc(epoch, source_name, acc_dict)

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

def random_sample(dataloader, fe, classifier, device, n):
    # Set the model to evaluation mode
    fe.eval()
    classifier.eval()

    all_results = []

    with torch.no_grad(): 
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = classifier(fe(inputs))
            
            probabilities = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)
            
            for i in range(inputs.size(0)):
                sample = inputs[i].tolist()  # Convert sample tensor to list
                prediction = preds[i].item()  # Get the prediction as a Python number
                confidence = max_probs[i].item()  # Get the confidence level as a Python number
                all_results.append((sample, prediction, confidence))

    # Randomly sample n% of the list
    num_samples = int(len(all_results) * n)
    sampled_results = random.sample(all_results, num_samples)

    # Separate features, labels, and confidences
    x = [item[0] for item in sampled_results]
    preds = [item[1] for item in sampled_results]
    confidences = [item[2] for item in sampled_results]

    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x)
    pred_tensor = torch.LongTensor(preds)
    confidences_tensor = torch.FloatTensor(confidences)

    subsample_dataset = TensorDataset(x_tensor, pred_tensor, confidences_tensor)

    return subsample_dataset

class RepeatSampler(torch.utils.data.Sampler):
    """ Sampler that repeats elements from a dataset. """
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return itertools.cycle(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)