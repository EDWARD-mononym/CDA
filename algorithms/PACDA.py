from scipy.spatial.distance import cdist
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
import os
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.prune as prune
import pickle
from algorithms.BaseAlgo import BaseAlgo
import numpy as np

#? https://github.com/PrasannaB29/PACDA/tree/master

class PACDA(BaseAlgo):
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

        self.mask = []
        self.hparams = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        loss_dict = defaultdict(float)
        if epoch == 0:
            self.pseudo_loader = self.obtain_label(trg_loader, device)
            self.prune_target()
        loss_dict = self.target_train(self.pseudo_loader, loss_dict, device)
        if epoch + 1 == self.hparams.n_epoch:
            loss_dict = self.finetune_target(self.pseudo_loader, loss_dict, device)
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

        self.prune_source()
        self.finetune_source(train_loader, device)

    def prune_source(self):
        mask_dict_F_w = {}
        mask_dict_F_b = {}
        for name, module in self.feature_extractor.named_modules():
            f=0
            if isinstance(module, torch.nn.Conv1d):
                prune.l1_unstructured(module, name='weight', amount=self.hparams.pf_c) # For unstructured pruning
                f=1
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.BatchNorm1d):
                prune.l1_unstructured(module, name='weight', amount=self.hparams.pf_bn)
                prune.l1_unstructured(module, name='bias', amount=self.hparams.pf_bn)
                f=2
            if f==1:
                mask_dict_F_w[name] = dict(module.named_buffers())['weight_mask']
                prune.remove(module, 'weight')
            elif f==2:
                mask_dict_F_w[name] = dict(module.named_buffers())['weight_mask']
                mask_dict_F_b[name] = dict(module.named_buffers())['bias_mask']
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')

        self.mask.append((mask_dict_F_w, mask_dict_F_b))

    def finetune_source(self, pseudo_loader, device):
        mask_dict_F_w, mask_dict_F_b = self.mask[0]

        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()
        self.classifier.train()

        for step, data in enumerate(pseudo_loader):
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)

            # Zero grads
            self.feature_extractor_optimiser.zero_grad()

            # Forward pass
            pred = self.classifier(self.feature_extractor(x))

            # Loss
            loss = self.taskloss(pred, y)
            loss.backward()

            for k, v in self.feature_extractor.named_parameters():
                if k.endswith("weight"):
                    tmp = k[:-7]  # because ".weight" has 7 characters
                    v.grad = torch.mul(v.grad, mask_dict_F_w[tmp])
                elif k.endswith("bias"):
                    tmp = k[:-5]
                    v.grad = torch.mul(v.grad, mask_dict_F_b[tmp])
                else:
                    print(" Should not happen "+k)

            # Step
            self.feature_extractor_optimiser.step()

    def prune_target(self):
        mask_dict_F_w_numpy, mask_dict_F_b_numpy = self.mask[-1]
        mask_prev_F_w, mask_prev_F_b = {}, {}
        tmp_mask_F_w, tmp_mask_F_b = {}, {}
        for key in mask_dict_F_w_numpy:
            tmp_mask_F_w[key] = 1e9 * mask_dict_F_w_numpy[key]
            mask_prev_F_w[key] = 1-mask_dict_F_w_numpy[key]
        for key in mask_dict_F_b_numpy:
            tmp_mask_F_b[key] = 1e9 * mask_dict_F_b_numpy[key]
            mask_prev_F_b[key] = 1-mask_dict_F_b_numpy[key]

        # Copy the network
        temp_fe = deepcopy(self.feature_extractor)
        temp_fe.eval()

        # Setting source params in temp_f and temp_b to 1e9
        with torch.no_grad():
            for k, v in temp_fe.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    tmp = k[:-7]  # because ".weight" has 7 characters
                    v.data = torch.mul(v.data, mask_prev_F_w[tmp]) + tmp_mask_F_w[tmp]

                elif (k.endswith("bias")) and ("bn" not in k):
                    tmp = k[:-5]
                    v.data = torch.mul(v.data, mask_prev_F_b[tmp]) + tmp_mask_F_b[tmp]

        # Pruning of temp_fe
        mask_target_F_w = {}
        mask_target_F_b = {}
        with torch.no_grad():
            for name, module in temp_fe.named_modules():
                f = 0
                if isinstance(module, torch.nn.Conv1d):
                    prune.l1_unstructured(module, name='weight', amount=self.hparams.pf_c)
                    f = 1
                # prune 40% of connections in all linear layers
                elif isinstance(module, torch.nn.BatchNorm1d):
                    # print("2 " + name)
                    prune.l1_unstructured(module, name='weight', amount=self.hparams.pf_bn)
                    prune.l1_unstructured(module, name='bias', amount=self.hparams.pf_bn)
                    f = 2
                if f == 1:
                    mask_target_F_w[name] = dict(module.named_buffers())['weight_mask']
                    prune.remove(module, 'weight')
                elif f == 2:
                    mask_target_F_w[name] = dict(module.named_buffers())['weight_mask']
                    mask_target_F_b[name] = dict(module.named_buffers())['bias_mask']
                    prune.remove(module, 'weight')
                    prune.remove(module, 'bias')

        # Mask for current target finetuning
        mask_cur_F_w, mask_cur_F_b = {}, {}

        for key in mask_prev_F_w:
            mask_cur_F_w[key] = torch.mul(mask_prev_F_w[key], mask_target_F_w[key])
        for key in mask_prev_F_b:
            mask_cur_F_b[key] = torch.mul(mask_prev_F_b[key], mask_target_F_b[key])
        
        # Save masks
        # mask_target_F_w is the mask used during eval. mask_cur_F_w is only for target_finetune hence need not be saved
        self.mask.append((mask_target_F_w, mask_target_F_b))

        # Deleting temporary model
        del temp_fe

        # Setting target model weights to pruned weights - starts
        with torch.no_grad():
            for k, v in self.feature_extractor.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    tmp = k[:-7]  # because ".weight" has 7 characters
                    v.data = torch.mul(v.data, mask_target_F_w[tmp])

                elif (k.endswith("bias")) and ("bn" not in k):
                    tmp = k[:-5]
                    v.data = torch.mul(v.data, mask_target_F_b[tmp])

        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        self.mask_cur_F_w, self.mask_cur_F_b = mask_cur_F_w, mask_cur_F_b

    def target_train(self, pseudo_loader, lostdict, device):
        # Load src mask
        mask_dict_F_w, mask_dict_F_b = self.mask[-1]

        self.feature_extractor.train()
        self.classifier.train()
        self.feature_extractor.to(device)
        self.classifier.to(device)

        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        for step, data in enumerate(pseudo_loader):
            x, pseudo_label = data[0].to(device), data[1].to(device)

            # Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            # Forward pass
            feat = self.feature_extractor(x)
            pred = self.classifier(feat)

            classifier_loss = torch.nn.CrossEntropyLoss()(pred, pseudo_label)

            softmax_out = torch.nn.Softmax(dim=1)(pred)
            entropy_loss = torch.mean(Entropy(softmax_out))

            loss = self.hparams.cls_wt * classifier_loss + self.hparams.ent_wt * entropy_loss
            loss.backward()

            with torch.no_grad():
                for k, v in self.feature_extractor.named_parameters():
                    if (k.endswith("weight")) and ("bn" not in k):
                        tmp = k[:-7]
                        v.grad = torch.mul(v.grad, mask_dict_F_w[tmp])
                        
                    elif (k.endswith("bias")) and ("bn" not in k):
                        tmp = k[:-5]
                        v.grad = torch.mul(v.grad, mask_dict_F_b[tmp])

            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()


            lostdict["avg_loss"] += loss.item() / len(x)
            lostdict["avg_cls_loss"] += classifier_loss.item() / len(x)
            lostdict["avg_ent_loss"] += entropy_loss.item() / len(x)

        return lostdict

    def finetune_target(self, trg_loader, lostdict, device):
        # Training starts
        for step, data in enumerate(trg_loader):
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)

            # Zero grads
            self.feature_extractor_optimiser.zero_grad()

            # Forward pass
            pred = self.classifier(self.feature_extractor(x))

            # Loss
            loss = self.taskloss(pred, y)
            loss.backward()

            with torch.no_grad():
                for k, v in self.feature_extractor.named_parameters():
                    if (k.endswith("weight")) and ("bn" not in k):
                        tmp = k[:-7]
                        v.grad = torch.mul(v.grad, self.mask_cur_F_w[tmp])
                    elif (k.endswith("bias")) and ("bn" not in k):
                        tmp = k[:-5]
                        v.grad = torch.mul(v.grad, self.mask_cur_F_b[tmp])

            # Step
            self.feature_extractor_optimiser.step()

            # * Log the losses
            lostdict["avg_loss"] += loss.item() / len(x)

        return lostdict

    def obtain_label(self, loader, device):
        all_fea = torch.tensor([]).float().cpu()
        all_output = torch.tensor([]).float().cpu()

        self.feature_extractor.to(device)
        self.classifier.to(device)

        with torch.no_grad():
            for step, data in enumerate(loader):
                inputs = data[0].to(device)

                feas = self.feature_extractor(inputs)
                outputs = self.classifier(feas)

                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

        all_output = torch.nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
        unknown_weight = 1 - ent / np.log(self.hparams.num_class)
        _, predict = torch.max(all_output, 1)

        # Normalize the features
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()
        aff = all_output.float().cpu().numpy()

        # Initialize cluster centers
        K = all_output.size(1)
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]

        # Calculate the distance between features and cluster centers
        dd = cdist(all_fea, initc[labelset], "cosine")
        # Assign each feature to the nearest cluster center
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        # Combine pseudo labels with trg loader
        samples, _ = zip(*[(data[0], data[1]) for data in loader.dataset])
        samples_tensor = torch.stack(samples)
        pseudo_dataset = torch.utils.data.TensorDataset(samples_tensor, torch.tensor(pred_label))
        pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=256, shuffle=True)
        return pseudo_loader

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 