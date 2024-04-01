from itertools import cycle
import os
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from algorithms.BaseAlgo import BaseAlgo
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
        self.taskloss = torch.nn.CrossEntropyLoss()
        
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
        pseudo_labels = self.obtain_label(trg_loader)

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
            # pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
            pseudo_label = torch.from_numpy(pseudo_labels[trg_idx.long()]).to(self.device)
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
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_ent_loss"] += entropy_loss.item() / len(trg_x)
            loss_dict["avg_pseud_target_loss"] += target_loss.item() / len(trg_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()

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
