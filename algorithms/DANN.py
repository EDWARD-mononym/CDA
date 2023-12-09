from collections import defaultdict
from itertools import cycle
import os
import torch
from torch.optim.lr_scheduler import StepLR
from algorithms.BaseAlgo import BaseAlgo
from architecture.Discriminator import Discriminator, ReverseLayerF
import numpy as np
#? https://github.com/fungtion/DANN/tree/master


class DANN(BaseAlgo):
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

        combined_loader = zip(cycle(src_loader), trg_loader)

        loss_dict = defaultdict(float)


        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

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

            domain_loss = src_domain_loss + self.hparams.da_wt * trg_domain_loss
            loss = src_cls_loss + domain_loss
            loss.backward()

            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()
            self.discriminator_optimiser.step()

            # * Log the losses
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_classification_loss"] += src_cls_loss.item() / len(src_x)
            loss_dict["avg_domain_loss"] += domain_loss.item() / len(trg_x)

            # * Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

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

