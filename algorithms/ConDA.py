from itertools import cycle
from collections import defaultdict
import itertools
import numpy as np
import os
import random
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo

class ConDA(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.hparam = configs
        self.buffer = None

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        # Freeze the classifier
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # send to device
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()

        # Append the target loader with the memory buffer
        if self.buffer:
            appended_dataset = ConcatDataset([trg_loader.dataset, self.buffer.dataset])
            appended_loader = DataLoader(appended_dataset, batch_size=trg_loader.batch_size, drop_last=False)
        else:
            appended_loader = trg_loader

        # obtain pseudo labels for each epoch
        pseudo_labels = self.obtain_label(appended_loader)

        loss_dict = defaultdict(float)

        combined_loader = zip(itertools.cycle(src_loader), appended_loader)
        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x, trg_idx = source[0].to(device), source[1].to(device), target[0].to(device), target[2]

            #* Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            #* Pseudo labeling by clustering
            pseudo_label = torch.from_numpy(pseudo_labels[trg_idx.long()]).to(device)

            #* Sample Mixup
            virtual_x, virtual_y = mixup_samples(trg_x, pseudo_label)

            #* Forward pass
            src_pred = self.classifier(self.feature_extractor(src_x))
            virtual_feat = self.feature_extractor(virtual_x)
            virtual_pred = self.classifier(virtual_feat)

            #* Src classification loss
            src_loss = self.taskloss(src_pred, src_y)

            #* Pseudo labeling loss
            mixup_loss = self.configs.mixup_loss_wt * F.cross_entropy(virtual_pred.squeeze(), virtual_y.long())

            #* Entropy loss
            softmax_out = nn.Softmax(dim=1)(virtual_pred)
            entropy_loss = self.configs.ent_loss_wt * torch.mean(self.EntropyLoss(softmax_out))

            #* Equal diversity loss
            eq_div_loss = self.configs.eq_div_loss_wt * -torch.sum(-softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

            # Total loss
            loss = src_loss + mixup_loss + entropy_loss + eq_div_loss

            #* Compute loss
            loss.backward()

            #* update weights
            self.feature_extractor_optimiser.step()

            # save average losses
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_mixup_loss"] += mixup_loss.item() / len(trg_x)
            loss_dict["avg_entropy_loss"] += entropy_loss.item() / len(trg_x)
            loss_dict["avg_eqdiv_loss"] += eq_div_loss.item() / len(trg_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()

        #* Save target to buffer
        if epoch == self.configs.n_epoch-1:
            subsampled_dataset = random_sample(trg_loader, self.feature_extractor, self.classifier, device, n = 0.01)
            repeated_sampler = RepeatSampler(subsampled_dataset)
            new_memory = DataLoader(subsampled_dataset, batch_size=trg_loader.batch_size, sampler=repeated_sampler, drop_last=False)

            # Update memory
            if self.buffer is None:
                self.buffer = new_memory
            else:
                concatenated_dataset = ConcatDataset([self.buffer.dataset, new_memory.dataset])
                repeated_sampler = RepeatSampler(concatenated_dataset)
                self.buffer = DataLoader(concatenated_dataset, batch_size=trg_loader.batch_size, sampler=repeated_sampler, drop_last=False)

        return loss_dict

    def obtain_label(self, loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        start_test = True
        with torch.no_grad():   
            for data in loader:
                inputs = data[0]
                inputs = inputs.cuda()
                labels = data[1]
                feas = self.feature_extractor(inputs)
                outputs = self.classifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-8), dim=1)
        # unknown_weight = 1 - ent / np.log(args.class_num)
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        print(log_str)

        self.feature_extractor.train()
        return predict.astype('int')

    def EntropyLoss(self, input_):
        mask = input_.ge(0.0000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = - (torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

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

def mixup_samples(samples, labels, alpha=1):
    B, C, L = samples.shape

    # Generate lambda from a Beta distribution
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(samples.device)

    # Randomly shuffle the samples and labels
    indices = torch.randperm(B).to(samples.device)
    shuffled_samples = samples[indices]
    shuffled_labels = labels[indices]

    # Perform mixup
    mixed_samples = lam.view(B, 1, 1) * samples + (1 - lam.view(B, 1, 1)) * shuffled_samples
    mixed_labels = lam * labels + (1 - lam) * shuffled_labels

    return mixed_samples, mixed_labels

