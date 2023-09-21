import importlib
import os
from pathlib import Path
import torch

from train.source_train import source_train

def load_source_model(configs, feature_extractor, classifier, scenario, device):
    folder_path = os.path.join(os.getcwd(), f"source_models/{configs['Dataset']['Dataset_Name']}/{configs['BackboneConfig']['Backbone']}")
    feature_extractor_path = os.path.join(folder_path, f"{scenario[0]}_feature.pt")
    classifier_path = os.path.join(folder_path, f"{scenario[0]}_classifier.pt")

    feature_extractor.load_state_dict(torch.load(feature_extractor_path))
    classifier.load_state_dict(torch.load(classifier_path))

    feature_extractor.to(device)
    classifier.to(device)

    return feature_extractor, classifier

def load_best_model(configs, feature_extractor, classifier, scenario, target_name, device):
    #* Load state dict
    save_folder = os.path.join(os.getcwd(), f'adapted_models/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}/{scenario}')
    feature_extractor.load_state_dict(torch.load(os.path.join(save_folder, f"{target_name}_feature.pt")))
    classifier.load_state_dict(torch.load(os.path.join(save_folder, f"{target_name}_classifier.pt")))

    feature_extractor.to(device)
    classifier.to(device)

    return feature_extractor, classifier