from itertools import cycle
import os
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from algorithms.BaseAlgo import BaseAlgo
from utils.model_testing import test_domain
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class SHOT(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)


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

        self.configs = configs
    def epoch_train(self, src_loader, trg_loader, epoch, device):
        # Freeze the classifier
        self.device = device
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # send to device
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()

        combined_loader = zip(cycle(src_loader), trg_loader)

        # obtain pseudo labels for each epoch
        pseudo_labels = self.obtain_pseudo_labels(trg_loader)

        loss_dict = defaultdict(float)

        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x, trg_idx = source[0], source[1], target[0], target[2]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            #* Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            #* Forward pass
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # pseudo labeling loss
            pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
            target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

            softmax_out = nn.Softmax(dim=1)(trg_pred)
            entropy_loss = self.configs.ent_loss_wt * torch.mean(self.EntropyLoss(softmax_out))

            #  Information maximization loss
            entropy_loss -= self.configs.im * torch.sum(-softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

            # Total loss
            loss = entropy_loss + self.configs.target_cls_wt * target_loss


            #* Compute loss
            loss.backward()

            #* update weights
            self.feature_extractor_optimiser.step()

            # save average losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_ent_loss"] += entropy_loss.item() / len(src_x)
            loss_dict["avg_pseud_target_loss"] += target_loss.item() / len(src_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        return loss_dict


    def pretrain(self, train_loader, test_loader, source_name, save_path, device):
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

                #* Zero grads
                self.feature_extractor_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Forward pass
                pred = self.classifier(self.feature_extractor(x))

                #* Loss
                loss = self.cross_entropy_label_smooth(pred, y,self.configs.num_class, device, epsilon=0.1)
                loss.backward()

                #* Step
                self.feature_extractor_optimiser.step()
                self.classifier_optimiser.step()

                running_loss += loss.item()

            #* Adjust learning rate
            self.fe_lr_scheduler.step()
            self.classifier_lr_scheduler.step()

            #* Save best model
            epoch_acc = test_domain(test_loader, self.feature_extractor, self.classifier, device)
            if epoch_acc > best_acc:
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))
            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
            print("-" * 30)  # Print a separator for clarity
    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)

                features = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)

        preds = torch.cat((preds))
        feas = torch.cat((feas))

        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)
        return pred_label

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
