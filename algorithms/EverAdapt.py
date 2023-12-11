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

class EverAdapt(BaseAlgo):
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
        self.taskloss = torch.nn.CrossEntropyLoss()

        # Added memory
        self.memory = None

    def epoch_train(self, src_loader, trg_loader, epoch, device):

        loss_dict = defaultdict(float)

        epoch_memory_inputs = []

        if self.memory:
            combined_loader = zip(itertools.cycle(src_loader), trg_loader, itertools.cycle(self.memory))
        else:
            combined_loader = zip(itertools.cycle(src_loader), trg_loader)

        for step, (data) in enumerate(combined_loader):
            if self.memory:
                source, target, memory = data
                mem_x = memory[0].to(device)
            else:
                source, target = data
                memory, mem_x = None, None

            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            #* Check if all are same size
            if memory:
                if mem_x.shape[0] != trg_x.shape[0]: # This happens towards the end where the remaining data in target is less than batchsize
                    continue

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate coral loss
            coral_loss = CORAL(src_feat, trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            alpha = 0.9 ** epoch # Give more priority to coral loss early on and more to lmmd later on

            loss = (1-alpha) * self.hparams.domain_loss_wt * domain_loss + alpha * self.hparams.src_cls_loss_wt * coral_loss + \
                self.hparams.src_cls_loss_wt * src_cls_loss 

            ###### ADDED REPLAY MEMORY #####
            if memory:
                # extract memory features
                mem_feat = self.feature_extractor(mem_x)
                mem_pred = self.classifier(mem_feat)

                # calculate memory lmmd loss
                mem_domain_loss = self.loss_LMMD.get_loss(src_feat, mem_feat, src_y, torch.nn.functional.softmax(mem_pred, dim=1))

                # calculate memory coral loss
                # mem_coral_loss = CORAL(src_feat, mem_feat)


                # loss += self.hparams.domain_loss_wt * mem_domain_loss + self.hparams.src_cls_loss_wt * mem_coral_loss
                loss += self.hparams.domain_loss_wt * mem_domain_loss
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

            #* Save target data
            epoch_memory_inputs.append(trg_x.cpu().detach())

        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        # Save target to memory
        if epoch == self.configs.n_epoch-1:
            n_save = 0.05
            # Select a portion of the current data for the memory
            indices = list(range(len(epoch_memory_inputs)))
            random.shuffle(indices)
            n_to_store = int(n_save * len(epoch_memory_inputs))

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