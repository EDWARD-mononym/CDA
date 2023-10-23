from collections import defaultdict
import importlib
from itertools import cycle
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo

#? https://arxiv.org/pdf/1712.02560.pdf

class MCD(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        classifier_name = configs.Classifier_Type
        imported_classifier = importlib.import_module(f"architecture.{classifier_name}")
        classifier_class = getattr(imported_classifier, classifier_name)

        self.classifier2 = classifier_class(configs)

        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
                # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
                # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=configs.step_size, gamma=configs.gamma)
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=configs.step_size, gamma=configs.gamma)
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=configs.step_size, gamma=configs.gamma)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.hparams = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.classifier2.to(device)
        self.feature_extractor.train()
        self.classifier.train()
        self.classifier2.train()

        combined_loader = zip(cycle(src_loader), trg_loader)

        loss_dict = defaultdict(float)

        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s = src_cls_loss1 + src_cls_loss2

            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x) 
            trg_pred1 = self.classifier(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())

            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis
            
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the classifiers
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
            # Unfreeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)        
            trg_pred1 = self.classifier(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)

            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams.domain_loss_wt * loss_dis_t 

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_trg_loss"] += loss_dis_t.item() / len(src_x)

        #* Adjust learning rate
        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()
        return loss_dict

    def pretrain(self, train_loader, test_loader, source_name, save_path, device, evaluator):
        best_acc = -1.0
        print(f"Training source model")
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}/{self.n_epoch}')

            self.feature_extractor.to(device)
            self.classifier.to(device)
            self.classifier2.to(device)
            self.feature_extractor.train()
            self.classifier.train()
            self.classifier2.train()
            running_loss = 0
            for step, data in enumerate(train_loader):
                x, y = data[0], data[1]
                x, y = x.to(device), y.to(device)

                src_feat = self.feature_extractor(x)
                src_pred1 = self.classifier(src_feat)
                src_pred2 = self.classifier2(src_feat)

                src_cls_loss1 = self.cross_entropy(src_pred1, y)
                src_cls_loss2 = self.cross_entropy(src_pred2, y)

                loss = src_cls_loss1 + src_cls_loss2

                self.optimizer_c1.zero_grad()
                self.optimizer_c2.zero_grad()
                self.optimizer_fe.zero_grad()

                loss.backward()

                self.optimizer_c1.step()
                self.optimizer_c2.step()
                self.optimizer_fe.step()

                running_loss += loss.item()

            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
            print("-" * 30)  # Print a separator for clarity

            #* Save best model
            epoch_acc = evaluator.test_domain(test_loader)
            if epoch_acc > best_acc:
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

    def discrepancy(self, out1, out2):
            return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))