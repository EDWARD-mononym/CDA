import importlib
import os
from pathlib import Path
import torch

from train.source_train import source_train

def load_source_model(configs, scenario, device):
    folder_path = os.path.join(os.getcwd(), f"source_models/{configs['Dataset']['Dataset_Name']}/{configs['BackboneConfig']['Backbone']}")
    feature_extractor_path = os.path.join(folder_path, f"{scenario[0]}_feature.pt")
    classifier_path = os.path.join(folder_path, f"{scenario[0]}_classifier.pt")

    #* Import the feature extractor & classifier 
    backbone_name, classifier_name = configs["BackboneConfig"]["Backbone"], configs["ClassifierConfig"]["Classifier"]
    imported_backbone = importlib.import_module(f"architecture.{backbone_name}")
    imported_classifier = importlib.import_module(f"architecture.{classifier_name}")
    backbone_class = getattr(imported_backbone, backbone_name)
    classifier_class = getattr(imported_classifier, classifier_name)

    feature_extractor = backbone_class(configs)
    classifier = classifier_class(configs)

    try:
        feature_extractor.load_state_dict(torch.load(feature_extractor_path))
        classifier.load_state_dict(torch.load(classifier_path))

    except FileNotFoundError:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        feature_extractor, classifier = source_train(feature_extractor, classifier, scenario[0], configs, folder_path, device)

    return feature_extractor, classifier

def load_best_model(configs, target_name):
    #* Import the feature extractor & classifier 
    backbone_name, classifier_name = configs["BackboneConfig"]["Backbone"], configs["ClassifierConfig"]["Classifier"]
    imported_backbone = importlib.import_module(f"architecture.{backbone_name}")
    imported_classifier = importlib.import_module(f"architecture.{classifier_name}")
    backbone_class = getattr(imported_backbone, backbone_name)
    classifier_class = getattr(imported_classifier, classifier_name)

    #* Create network
    feature_extractor = backbone_class(configs)
    classifier = classifier_class(configs)

    #* Load state dict
    save_folder = os.path.join(os.getcwd(), f'adapted_models/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}')
    feature_extractor.load_state_dict(torch.load(os.path.join(save_folder, f"{target_name}_feature.pt")))
    classifier.load_state_dict(torch.load(os.path.join(save_folder, f"{target_name}_classifier.pt")))

    return feature_extractor, classifier