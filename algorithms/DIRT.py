from collections import defaultdict
import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from algorithms.BaseAlgo import BaseAlgo
from architecture.Discriminator import Discriminator

class DIRT(BaseAlgo):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

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

        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss()
        self.vat_loss = VAT(self.network, device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)
        self.cross_entropy = nn.CrossEntropyLoss()

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )

        # hparams
        self.hparams = configs
       
    def epoch_train(self, src_loader, trg_loader, epoch, device):

        # Construct Joint Loaders 
        combined_loader = zip(src_loader, itertools.cycle(trg_loader))

        loss_dict = defaultdict(float)

        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(device)
            domain_label_trg = torch.zeros(len(trg_x)).to(device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat

            # total loss
            loss = self.hparams.src_cls_loss_wt * src_cls_loss + self.hparams.domain_loss_wt * domain_loss + \
                self.hparams.cond_ent_wt * loss_trg_cent + self.hparams.vat_loss_wt * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            loss.backward()
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(src_x)
            loss_dict["avg_src_cls_loss"] += src_cls_loss.item() / len(src_x)
            loss_dict["avg_loss_trg_cent"] += loss_trg_cent.item() / len(src_x)
            loss_dict["avg_total_vat"] += total_vat.item() / len(src_x)

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

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class VAT(nn.Module):
    def __init__(self, model, device):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5
        self.device = device

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device=self.device)

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]