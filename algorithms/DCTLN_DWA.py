from collections import defaultdict
import copy
import itertools
import os
import random
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from algorithms.BaseAlgo import BaseAlgo
from architecture.Discriminator import Discriminator, ReverseLayerF
import numpy as np



class DCTLN_DWA(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.discriminator = Discriminator(configs)
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
        self.discriminator_optimiser = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser,
                                      step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser,
                                              step_size=configs.step_size, gamma=configs.gamma)

        self.source_feature_extractor = None
        self.source_classifier = None
        self.memory = None
        
        self.hparams = configs


    def epoch_train(self, src_loader, trg_loader, epoch, device):
        # Sending models to GPU
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.discriminator.to(device)

        # Make models to be
        self.feature_extractor.train()
        self.classifier.train()
        self.discriminator.train()
        self.source_feature_extractor.eval()
        self.source_classifier.eval()

        combined_loader = zip(itertools.cycle(src_loader), trg_loader, itertools.cycle(self.memory))

        loss_dict = defaultdict(float)
        for step, (source, target, memory) in enumerate(combined_loader):
            src_x, src_y, trg_x, mem_x = source[0].to(device), source[1].to(device), target[0].to(device), memory[0].to(device)

            # * Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.discriminator_optimiser.zero_grad()

            p = float(step + epoch * len(trg_loader)) / self.hparams.n_epoch / len(trg_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_domain_label = torch.zeros(len(src_x)).long().cuda()
            trg_domain_labels = torch.ones(len(trg_x)).long().cuda()

            #* Forward pass
            src_feature = self.feature_extractor(src_x)
            src_output = self.classifier(src_feature)
            trg_feat = self.feature_extractor(trg_x)
            mem_pred = self.classifier(self.feature_extractor(mem_x))
            with torch.no_grad():
                mem_pred_src_model = self.source_classifier(self.source_feature_extractor(mem_x))

            #* Reserved knowledge loss
            rk_loss = self.reserved_knowledge_loss(mem_pred, mem_pred_src_model)

            #* Task classification
            src_cls_loss = self.taskloss(src_output.squeeze(), src_y)

            #* Domain classification
            # Source
            src_feat_reversed = ReverseLayerF.apply(src_feature, alpha)
            src_domain_pred = self.discriminator(src_feat_reversed)
            src_domain_loss = self.taskloss(src_domain_pred, src_domain_label)

            # Target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.discriminator(trg_feat_reversed)
            trg_domain_loss = self.taskloss(trg_domain_pred, trg_domain_labels)

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams.src_wt * src_cls_loss + self.hparams.da_wt * domain_loss + self.hparams.rk_wt * rk_loss
            loss.backward()

            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()
            self.discriminator_optimiser.step()

            # * Log the losses
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_src_loss"] += src_cls_loss.item() / len(src_x)
            loss_dict["avg_domain_loss"] += domain_loss.item() / len(trg_x)
            loss_dict["avg_rk_loss"] += rk_loss.item() / len(trg_x)

        # * Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        # Save target to memory
        if epoch == self.configs.n_epoch-1:
            #* Save source model
            self.source_feature_extractor = copy.deepcopy(self.feature_extractor)
            self.source_classifier = copy.deepcopy(self.classifier)

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

            #* Save best model
            acc_dict = evaluator.test_all_domain(self)
            epoch_acc = acc_dict[source_name]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epoch ACC: {acc_dict[source_name]}")
            print("-" * 30)  # Print a separator for clarity

            #* Log epoch acc
            evaluator.update_epoch_acc(epoch, source_name, acc_dict)

        #* Save source model
        self.source_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.source_classifier = copy.deepcopy(self.classifier)

        #* Save source data into memory
        subsampled_dataset = random_sample(train_loader, self.feature_extractor, self.classifier, device, n = 0.01)
        repeated_sampler = RepeatSampler(subsampled_dataset)
        new_memory = DataLoader(subsampled_dataset, batch_size=train_loader.batch_size, sampler=repeated_sampler, drop_last=False)
        # Update memory
        if self.memory is None:
            self.memory = new_memory
        else:
            concatenated_dataset = ConcatDataset([self.memory.dataset, new_memory.dataset])
            repeated_sampler = RepeatSampler(concatenated_dataset)
            self.memory = DataLoader(concatenated_dataset, batch_size=train_loader.batch_size, sampler=repeated_sampler, drop_last=False)

    def reserved_knowledge_loss(self, teacher_logits, student_logits, T=2):
        #Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

        return torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

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



