from itertools import cycle
import numpy as np
import os
import torch
from torch.autograd import Function
from torch import nn

from utils.model_testing import test_all_domain

#? https://github.com/fungtion/DANN/tree/master

def DANN(src_loader, trg_loader, feature_extractor, classifier, discriminator,
         feature_extractor_optimiser,  classifier_optimiser, discriminator_optimiser, fe_lr_scheduler, classifier_lr_scheduler,
         n_epoch, save_path, target_name, device, datasetname, scenario, writer):
    best_acc = -1.0

    print(f"Adapting to {target_name}")
    for epoch in range(n_epoch):
        print(f"Epoch: {epoch}/{n_epoch}")

        epoch_train(src_loader, trg_loader, feature_extractor, classifier, discriminator,
                    feature_extractor_optimiser, classifier_optimiser, discriminator_optimiser,
                    epoch, n_epoch, device)

        # Test & Save best model
        acc_dict = test_all_domain(datasetname, scenario, feature_extractor, classifier, device)

        if acc_dict[target_name] > best_acc:
            torch.save(feature_extractor.state_dict(), os.path.join(save_path, f"{target_name}_feature.pt"))
            torch.save(classifier.state_dict(), os.path.join(save_path, f"{target_name}_classifier.pt"))

        # Log the accuracy of each epoch
        for domain in acc_dict:
            writer.add_scalar(f'Acc/{domain}', acc_dict[domain], epoch)

        # Adjust Learning Rate
        fe_lr_scheduler.step()
        classifier_lr_scheduler.step()

def epoch_train(src_loader, trg_loader, feature_extractor, classifier, discriminator,
                feature_extractor_optimiser,  classifier_optimiser, discriminator_optimiser, 
                epoch, n_epoch, device):
    feature_extractor.train()
    classifier.train()
    discriminator.train()
    combined_loader = zip(cycle(src_loader), trg_loader)

    for step, (source, target) in enumerate(combined_loader):
        src_x, src_y, trg_x = source[0], source[1], target[0]
        src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

        p = float(step + epoch * len(trg_loader)) / n_epoch / len(trg_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        src_domain_label = torch.zeros(len(src_x)).long().cuda()
        trg_domain_labels = torch.ones(len(trg_x)).long().cuda()

        #* Zero grads
        feature_extractor_optimiser.zero_grad()
        classifier_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        #* Forward pass
        src_feature = feature_extractor(src_x)
        src_output = classifier(src_feature)
        trg_feat = feature_extractor(trg_x)

        #* Task classification
        src_cls_loss = loss_class(src_output.squeeze(), src_y)

        #* Domain classification
        # Source
        src_feat_reversed = ReverseLayerF.apply(src_feature, alpha)
        src_domain_pred = discriminator(src_feat_reversed)
        src_domain_loss = loss_domain(src_domain_pred, src_domain_label)

        # Target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = discriminator(trg_feat_reversed)
        trg_domain_loss = loss_domain(trg_domain_pred, trg_domain_labels)

        domain_loss = src_domain_loss + trg_domain_loss
        loss = src_cls_loss + domain_loss
        loss.backward()

        feature_extractor_optimiser.step()
        classifier_optimiser.step()
        discriminator_optimiser.step()

loss_class = torch.nn.NLLLoss().cuda()
loss_domain = torch.nn.NLLLoss().cuda()

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs["ClassifierConfig"]["input_size"], 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None