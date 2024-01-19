import csv
from collections import defaultdict
import itertools
from itertools import cycle
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from algorithms.BaseAlgo import BaseAlgo
from utils.avg_meter import AverageMeter

class DSAN(BaseAlgo):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

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
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_class).to(device)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.cond_ent = ConditionalEntropyLoss()

        ###### ADDED MEMORY #######
        self.memory = None

    def epoch_train(self, src_loader, trg_loader, epoch, device):

        #! TESTING PSEUDO LABEL ACC
        correct, total = 0, 0

        # Construct Joint Loaders 
        combined_loader = zip(src_loader, itertools.cycle(trg_loader))

        loss_dict = defaultdict(float)

        epoch_memory_inputs = []

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

            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            if memory:
                if mem_x.shape[0] != trg_x.shape[0]: # This happens towards the end where the remaining data in target is less than batchsize
                    continue

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            #! Testing pseudo label
            trg_y = target[1].to(device)
            _, pseudo_label = torch.max(trg_pred, 1)
            batch_total = trg_y.size(0)
            total += batch_total
            batch_correct = (pseudo_label == trg_y).sum().item()
            correct += batch_correct
            #! Testing pseudo labels
            batch_accuracy = batch_correct / batch_total
            batch_csv_file = f"DSAN/epoch_{epoch}.csv"
            with open(batch_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step, epoch, batch_accuracy])

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            #! ADDED CORAL LOSS
            # coral_loss = CORAL(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            # calculate the total loss
            loss = self.hparams.domain_loss_wt * domain_loss + \
                self.hparams.src_cls_loss_wt * src_cls_loss + \
                cond_ent_loss

            ###### ADDED REPLAY MEMORY #####
            if memory:
                mem_feat = self.feature_extractor(mem_x)
                mem_pred = self.classifier(mem_feat)

                # calculate lmmd loss
                mem_loss = self.loss_LMMD.get_loss(src_feat, mem_feat, src_y, torch.nn.functional.softmax(mem_pred, dim=1))

                #! ADDED CORAL LOSS
                # coral_loss = CORAL(src_feat, mem_feat)
                mem_cond_ent_loss = self.cond_ent(mem_feat)

                loss += self.hparams.domain_loss_wt * mem_loss
                loss += mem_cond_ent_loss
             ####### END OF REPLAY MEMORY SECTION #####

            # update feature extractor
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            loss.backward()
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_src_cls_loss"] += src_cls_loss.item() / len(src_x)

            epoch_memory_inputs.append(trg_x.cpu().detach())

        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        # Memory
        if epoch == self.configs.n_epoch-1:
            # Select a portion of the current data for the memory
            indices = list(range(len(epoch_memory_inputs)))
            random.shuffle(indices)
            # n_to_store = int(self.configs.alpha * len(epoch_memory_inputs))
            n_to_store = int(0.10 * len(epoch_memory_inputs))
            selected_indices = indices[:n_to_store]

            selected_inputs = [epoch_memory_inputs[i] for i in selected_indices]
            
            selected_inputs = torch.cat(selected_inputs)

            new_memory = DataLoader(TensorDataset(selected_inputs), batch_size=trg_loader.batch_size, shuffle=True)

            # Update memory
            if self.memory is None:
                self.memory = new_memory
            else:
                concatenated_dataset = ConcatDataset([self.memory.dataset, new_memory.dataset])
                self.memory = DataLoader(concatenated_dataset, batch_size=trg_loader.batch_size)

        accuracy = correct / total
        csv_file = "DSAN_epoch_acc.csv"
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, accuracy])

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
                loss = self.cross_entropy(pred, y)
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

class LMMD_loss(nn.Module):
    def __init__(self, device, class_num=3, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.device = device

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)

        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=4):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
    
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

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)