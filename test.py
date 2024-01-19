import os
import torch

from architecture.CNN import CNN
from architecture.Classifier import Classifier
from configs.FD.DeepCORAL import configs
from dataloaders.FD import PB_Artificial_test

def test_domain_per_class(test_loader, feature_extractor, classifier, device):
    feature_extractor.eval()
    classifier.eval()
    
    # Initialize variables to track correct predictions and total number of samples for each class
    correct_per_class = {0: 0, 1: 0, 2: 0}
    total_per_class = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)
            logits = classifier(feature_extractor(x))
            _, pred = torch.max(logits, 1)
            
            for label in range(3):  # Assuming there are 3 classes labeled as 0, 1, and 2
                # Update total count for each class
                total_per_class[label] += (y == label).sum().item()
                
                # Update correct count for each class
                correct_per_class[label] += ((pred == label) & (y == label)).sum().item()
                
    # Compute per-class accuracy
    accuracy_per_class = {label: correct_per_class[label] / total_per_class[label] for label in range(3)}
    
    return accuracy_per_class

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def main():
    feature_extractor, classifier = CNN(configs), Classifier(configs)

    save_folder = os.path.join(os.getcwd(), "adapted_models/FD/DeepCORAL/('CWRU_DE', 'PB_Artificial', 'PB_Real', 'CWRU_FE')")

    feature_extractor.load_state_dict(torch.load(os.path.join(save_folder, "PB_Artificial_feature.pt")))
    classifier.load_state_dict(torch.load(os.path.join(save_folder, "PB_Artificial_classifier.pt")))

    feature_extractor, classifier = feature_extractor.to(device), classifier.to(device)

    acc = test_domain_per_class(PB_Artificial_test, feature_extractor, classifier, device)

    print(acc)

if __name__ == "__main__":
    main()