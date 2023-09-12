from itertools import cycle
import os
import torch

from utils.model_testing import test_all_domain

#? https://arxiv.org/abs/1607.01719

def DeepCORAL(src_loader, trg_loader, feature_extractor, classifier,
              feature_extractor_optimiser,  classifier_optimiser, n_epoch, save_path, target_name, device, datasetname, scenario, writer):
    best_acc = 0.0

    print(f"Adapting to {target_name}")
    for epoch in range(n_epoch):
        print(f"Epoch: {epoch}/{n_epoch}")
        # Adaptation
        epoch_train(src_loader, trg_loader, feature_extractor, classifier,
              feature_extractor_optimiser, classifier_optimiser, device)

        # Test & Save best model
        acc_dict = test_all_domain(datasetname, scenario, feature_extractor, classifier, device)
        print(acc_dict)

        if acc_dict[target_name] > best_acc:
            torch.save(feature_extractor.state_dict(), os.path.join(save_path, f"{target_name}_feature.pt"))
            torch.save(classifier.state_dict(), os.path.join(save_path, f"{target_name}_classifier.pt"))

        # Log the accuracy of each epoch
        for domain in acc_dict:
            writer.add_scalar(f'Acc/{domain}', acc_dict[domain], epoch)

def epoch_train(src_loader, trg_loader, feature_extractor, classifier,
              feature_extractor_optimiser, classifier_optimiser, device):
    feature_extractor.train()
    classifier.train()

    combined_loader = zip(cycle(src_loader), trg_loader)

    for step, (source, target) in enumerate(combined_loader):
        src_x, src_y, trg_x = source[0], source[1], target[0]
        src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

        #* Zero grads
        feature_extractor_optimiser.zero_grad()
        classifier_optimiser.zero_grad()

        #* Forward pass
        src_feat = feature_extractor(src_x)
        src_pred = classifier(src_feat)
        trg_feat = feature_extractor(trg_x)

        #* Compute loss
        classification_loss = torch.nn.functional.cross_entropy(src_pred, src_y)
        coral_loss = CORAL(src_feat, trg_feat)
        loss = classification_loss + coral_loss
        loss.backward()

        #* Step
        feature_extractor_optimiser.step()
        classifier_optimiser.step()

def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss