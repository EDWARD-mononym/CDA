from itertools import cycle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from algorithms.BaseAlgo import BaseAlgo
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import numpy as np
import importlib
import random
from utils.create_logger import AverageMeter
from collections import defaultdict

class COSDA(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        backbone_name, classifier_name = configs.Backbone_Type, configs.Classifier_Type
        imported_backbone = importlib.import_module(f"architecture.{backbone_name}")
        imported_classifier = importlib.import_module(f"architecture.{classifier_name}")
        backbone_class = getattr(imported_backbone, backbone_name)
        classifier_class = getattr(imported_classifier, classifier_name)

        self.teacher_backbone = backbone_class(configs)
        self.teacher_classifier = classifier_class(configs)

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

        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser,   step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser,     step_size=configs.step_size, gamma=configs.gamma)

        self.memory = None
        self.configs = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        utilized_ratio = AverageMeter()

        # send to device
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.teacher_backbone.to(device)
        self.teacher_classifier.to(device)
        # Train
        self.feature_extractor.train()
        self.classifier.train()
        self.teacher_backbone.train()
        self.teacher_classifier.train()

        loss_dict = defaultdict(float)

        epoch_memory_inputs, epoch_memory_labels = [], []

        if self.memory:
            combined_loader = zip(cycle(src_loader), trg_loader, cycle(self.memory))
        else:
            combined_loader = zip(cycle(src_loader), trg_loader)

        for step, (data) in enumerate(combined_loader):
            if self.memory:
                source, target, memory = data
                mem_x = memory[0].to(device)
            else:
                source, target = data
                memory, mem_x = None, None

            # Extract data from source and target batches
            src_x, src_y, trg_x, trg_y = source[0], source[1], target[0], target[1]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            # Reset the gradients
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            # Obtain teacher's predictions without gradients
            with torch.no_grad():
                predictions = self.teacher_classifier(self.teacher_backbone(trg_x))
                knowledge, knowledge_mask = self.distill_knowledge(predictions, self.configs.conf_gate,
                                                                   temperature= self.configs.temp)

            # Mixup preparation
            if  self.configs.beta > 0:
                lam = np.random.beta( self.configs.beta, self.configs.beta)
                lam = max(lam, 1 - lam)  # Ensure lam is closer to 1
            else:
                lam = 1

            batch_size = trg_x.size(0)
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * trg_x + (1 - lam) * trg_x[index, :]
            mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]

            # Compute consistency loss
            mixed_output = self.classifier(self.feature_extractor(mixed_image))
            mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)
            consistency_loss = torch.sum(
                knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)
            ) / torch.sum(knowledge_mask) #! Potential to crash?

            # Compute regularization
            output = self.classifier(self.feature_extractor(trg_x))
            softmax_output = torch.softmax(output, dim=1)
            margin_output = torch.mean(softmax_output, dim=0)
            log_softmax_output = torch.log_softmax(output, dim=1)
            log_margin_output = torch.log(margin_output + 1e-5)

            mutual_info_loss = -1 * torch.mean(
                torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1)
            )


            ###### ADDED REPLAY MEMORY #####
            if memory:
                # Obtain teacher's predictions without gradients
                with torch.no_grad():
                    predictions = self.teacher_classifier(self.teacher_backbone(mem_x))
                    knowledge, knowledge_mask = self.distill_knowledge(predictions, self.configs.conf_gate,
                                                                    temperature= self.configs.temp)

                # Mixup preparation
                if  self.configs.beta > 0:
                    lam = np.random.beta( self.configs.beta, self.configs.beta)
                    lam = max(lam, 1 - lam)  # Ensure lam is closer to 1
                else:
                    lam = 1

                batch_size = mem_x.size(0)
                index = torch.randperm(batch_size).cuda()
                mixed_image = lam * mem_x + (1 - lam) * mem_x[index, :]
                mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]

                # Compute consistency loss
                mixed_output = self.classifier(self.feature_extractor(mixed_image))
                mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)
                mem_consistency_loss = torch.sum(
                    knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)
                ) / torch.sum(knowledge_mask) #! Potential to crash?

                # Compute regularization
                output = self.classifier(self.feature_extractor(mem_x))
                softmax_output = torch.softmax(output, dim=1)
                margin_output = torch.mean(softmax_output, dim=0)
                log_softmax_output = torch.log_softmax(output, dim=1)
                log_margin_output = torch.log(margin_output + 1e-5)

                mem_mutual_info_loss = -1 * torch.mean(
                    torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1)
                )

                consistency_loss += mem_consistency_loss
                mutual_info_loss += mem_mutual_info_loss
            ####### END OF REPLAY MEMORY SECTION #####

            # Calculate final task loss
            if  self.configs.only_mi:
                task_loss =  self.configs.reg_alpha * mutual_info_loss
            else:
                task_loss = consistency_loss +  self.configs.reg_alpha * mutual_info_loss

            # Backward propagation and optimizer steps
            task_loss.backward()
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()  # There seems to be a typo in the original code; it should be 'optimizer' and not 'optimiser'

            # Compute ratio of knowledge utilization
            ratio = float(torch.mean(knowledge_mask))
            utilized_ratio.update(ratio, knowledge_mask.size(0))


            loss_dict["avg_task_loss"] += task_loss.item() / len(src_x)
            loss_dict["avg_mutual_info_loss"] += mutual_info_loss.item() / len(src_x)
            loss_dict["avg_consistency_loss"] += consistency_loss.item() / len(src_x)

            epoch_memory_inputs.append(trg_x.cpu().detach())
            epoch_memory_labels.append(trg_y.cpu().detach())

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

    def pretrain(self, train_loader, test_loader, source_name, save_path, device, evaluator):
        best_acc = -1.0
        print(f"Training source model")
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}/{self.n_epoch}')

            self.feature_extractor.to(device)
            self.classifier.to(device)
            self.feature_extractor.train()
            self.classifier.train()

            for step, data in enumerate(train_loader):
                x, y = data[0], data[1]
                x, y = x.to(device), y.to(device)

                #* Zero grads
                self.feature_extractor_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Forward pass
                pred = self.classifier(self.feature_extractor(x))

                #* Loss
                loss = self.cross_entropy_label_smooth(pred, y, self.configs.num_class, device, epsilon=0.1)
                loss.backward()

                #* Step
                self.feature_extractor_optimiser.step()
                self.classifier_optimiser.step()

            #* Adjust learning rate
            self.fe_lr_scheduler.step()
            self.classifier_lr_scheduler.step()

            #* Save best model
            epoch_acc = evaluator.test_domain(self, test_loader)
            if epoch_acc > best_acc:
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))
        self.teacher_backbone.load_state_dict(self.feature_extractor.state_dict())
        self.teacher_classifier.load_state_dict(self.classifier.state_dict())
    def distill_knowledge(self, score, confidence_gate, temperature=0.07):
        predict = torch.softmax(score, dim=1)
        # get the knowledge with weight and mask
        max_p, max_p_class = predict.max(1)
        knowledge_mask = (max_p > confidence_gate).float().cuda()
        knowledge = torch.softmax(score / temperature, dim=1)
        return knowledge, knowledge_mask

    def cross_entropy_label_smooth(self, inputs, targets, num_classes, device, epsilon=0.1):
        logsoftmax = nn.LogSoftmax(dim=1)

        log_probs = logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).to(device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes

        loss = (- targets * log_probs).mean(0).sum()

        return loss

    def EntropyLoss(self, input_):
        mask = input_.ge(0.0000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = - (torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))
