import os
import torch

from utils.get_loaders import get_loader
from utils.model_testing import test_domain

def source_train(feature_extractor, classifier, source_name, configs, save_path, device):
    #* Load data & define optimisers
    train_loader = get_loader(configs["Dataset"]["Dataset_Name"], source_name, "train")
    test_loader = get_loader(configs["Dataset"]["Dataset_Name"], source_name, "test") # To save the best model

    feature_extractor_optimiser = torch.optim.SGD(feature_extractor.parameters(), 
                                                  lr=configs["OptimiserConfig"]["lr"],
                                                  momentum=configs["OptimiserConfig"]["momentum"])
    classifier_optimiser = torch.optim.SGD(classifier.parameters(), 
                                           lr=configs["OptimiserConfig"]["lr"],
                                           momentum=configs["OptimiserConfig"]["momentum"])
    task_loss = torch.nn.CrossEntropyLoss()

    #* Training
    best_acc = 0
    print(f"Training source model (source: {source_name})")

    feature_extractor.to(device)
    classifier.to(device)

    for epoch in range(configs["TrainingConfigs"]["n_epoch"]):
        print(f'Epoch: {epoch}/{configs["TrainingConfigs"]["n_epoch"]}')

        feature_extractor.train()
        classifier.train()

        for step, data in enumerate(train_loader):
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)

            #* Zero grads
            feature_extractor_optimiser.zero_grad()
            classifier_optimiser.zero_grad()

            #* Forward pass
            pred = classifier(feature_extractor(x))

            #* Loss
            loss = task_loss(pred, y)
            loss.backward()

            #* Step
            feature_extractor_optimiser.step()
            classifier_optimiser.step()

        #* Save best model
        epoch_acc = test_domain(test_loader, feature_extractor, classifier, device)
        if epoch_acc > best_acc:
            torch.save(feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
            torch.save(classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

    return feature_extractor, classifier