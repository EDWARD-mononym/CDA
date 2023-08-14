from collections import defaultdict
import numpy as np
import os
import torch
import torch.nn.functional as F

from algorithms.BaseModel import BaseModel
from architecture.discriminator import Discriminator
from architecture.MLP import MLP

class MMDA(BaseModel):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, configs):
        super().__init__(configs)

        # Aligment losses
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

    def update(self, src_loader, trg_loader, target_id, save_path, test_loader=None):

        best_loss = float('inf')
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.train_params["N_epochs"]):
            #* Set to train
            self.feature_extractor.train()
            self.classifier.train()

            # Construct Joint Loaders 
            joint_loader = enumerate(zip(src_loader, trg_loader))

            for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
                losses = defaultdict(float) #* To record losses
                src_x, src_y, trg_x = src_x.to(self.configs.device), src_y.to(self.configs.device), trg_x.to(self.configs.device)

                #* Forward pass
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                #* Source classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                #* Forward pass
                trg_feat = self.feature_extractor(trg_x)
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                src_cls_loss = self.cross_entropy(src_pred, src_y)

                trg_feat = self.feature_extractor(trg_x)

                coral_loss = self.coral(src_feat, trg_feat)
                mmd_loss = self.mmd(src_feat, trg_feat)
                cond_ent_loss = self.cond_ent(trg_feat)

                loss = self.configs.alg_hparams["coral_wt"] * coral_loss + \
                    self.configs.alg_hparams["mmd_wt"] * mmd_loss + \
                    self.configs.alg_hparams["cond_ent_wt"] * cond_ent_loss + \
                    self.configs.alg_hparams["src_cls_loss_wt"] * src_cls_loss

                #* Zero gradient
                self.feature_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Calculate Loss
                loss.backward()

                #* Backward propagation
                self.feature_optimiser.step()
                self.classifier_optimiser.step()

                # losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                #         'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
                
                losses["loss"] += loss.item() / len(src_loader)

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"feature_extractor_{target_id}.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"classifier_{target_id}.pt"))

            #* If the test_loader was given, test the performance of current epoch on the test domain
            if test_loader and (epoch+1) % 10 == 0:
                self.evaluate(test_loader, epoch, target_id)

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

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

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

class CORAL(torch.nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

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

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)