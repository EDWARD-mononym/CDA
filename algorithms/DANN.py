from itertools import cycle
import numpy as np
import os
import torch
from torch.autograd import Function
from torch import nn

from utils.model_testing import test_all_domain

#? https://github.com/fungtion/DANN/tree/master

def DANN(src_loader, trg_loader, feature_extractor, classifier, discriminator,
         feature_extractor_optimiser,  classifier_optimiser, discriminator_optimiser,
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

def epoch_train(src_loader, trg_loader, feature_extractor, classifier, discriminator,
                feature_extractor_optimiser,  classifier_optimiser, discriminator_optimiser, 
                epoch, n_epoch, device):
    feature_extractor.train()
    classifier.train()
    discriminator.train()
    combined_loader = zip(cycle(src_loader), trg_loader)

    for step, (source, target) in combined_loader:
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

        #* Source
        src_feature = feature_extractor(src_x)
        src_reverse_feature = feature_extractor.apply(src_feature, alpha)
        src_output = classifier(src_feature)
        src_domain_output = discriminator(src_reverse_feature)

        src_classification_loss = loss_class(src_output, src_y)
        src_domain_loss = loss_domain(src_domain_output, src_domain_label)

        #* Target
        trg_domain_output = discriminator(ReverseLayerF.apply(feature_extractor(trg_x), alpha))
        trg_domain_loss = loss_domain(trg_domain_output, trg_domain_labels)

        loss = src_classification_loss + src_domain_loss + trg_domain_loss
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