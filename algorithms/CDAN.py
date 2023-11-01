from collections import defaultdict
from itertools import cycle
import math
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo

#? https://arxiv.org/abs/1705.10667

class CDAN(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.feature_length * configs.output_channels, configs.num_class],
                                        configs.feature_length * configs.output_channels)

        # optimizer and scheduler
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
        self.domain_classifier_optimiser = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser,
                                      step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser,
                                              step_size=configs.step_size, gamma=configs.gamma)

        self.criterion_cond = ConditionalEntropyLoss()
        self.taskloss = torch.nn.CrossEntropyLoss()
        self.hparams = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.domain_classifier.to(device)
        self.feature_extractor.train()
        self.classifier.train()
        self.domain_classifier.train()

        self.criterion_cond.to(device)

        combined_loader = zip(cycle(src_loader), trg_loader)

        loss_dict = defaultdict(float)

        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(device)
            domain_label_trg = torch.zeros(len(trg_x)).to(device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.taskloss(disc_prediction, domain_label_concat)
            # update Domain classification
            self.domain_classifier_optimiser.zero_grad()
            disc_loss.backward()
            self.domain_classifier_optimiser.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss = self.taskloss(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.taskloss(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams.src_wt * src_cls_loss + self.hparams.domain_wt * domain_loss + self.hparams.cond_wt * loss_trg_cent
            
            # * Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_classification_loss"] += src_cls_loss.item() / len(src_x)
            loss_dict["avg_domain_loss"] += domain_loss.item() / len(src_x)
            loss_dict["avg_cond_loss"]+= loss_trg_cent.item() / len(src_x)

            
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

        # * Save best model
        epoch_acc = evaluator.test_domain(self, test_loader)
        if epoch_acc > best_acc:
            torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
            torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

class Discriminator_CDAN(torch.nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(configs.feature_length * configs.output_channels * configs.num_class, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class RandomLayer(torch.nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]