import importlib
import os
from pathlib import Path
import torch

from utils.get_loaders import get_loader

def adapt(feature_extractor, classifier, target_name, scenario, configs, device, writer):
    #* Load data & define optimisers
    train_trg_loader = get_loader(configs["Dataset"]["Dataset_Name"], target_name, "train")
    test_trg_loader = get_loader(configs["Dataset"]["Dataset_Name"], target_name, "test")
    src_loader = get_loader(configs["Dataset"]["Dataset_Name"], scenario[0], "train")

    feature_extractor_optimiser = torch.optim.SGD(feature_extractor.parameters(), 
                                                  lr=configs["OptimiserConfig"]["lr"],
                                                  momentum=configs["OptimiserConfig"]["momentum"])
    classifier_optimiser = torch.optim.SGD(classifier.parameters(), 
                                           lr=configs["OptimiserConfig"]["lr"],
                                           momentum=configs["OptimiserConfig"]["momentum"])

    save_folder = os.path.join(os.getcwd(), f'adapted_models/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]/scenario}')
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    #* Adaptation
    if configs["AdaptationConfig"]["Method"] == "DeepCORAL":
        imported_algo = importlib.import_module(f"algorithms.DeepCORAL")
        DeepCORAL = getattr(imported_algo, "DeepCORAL") #? These two lines are equivalent to "from algorithms.DeepCORAL import DeepCORAL"
        DeepCORAL(src_loader, train_trg_loader, 
            feature_extractor, classifier, feature_extractor_optimiser, classifier_optimiser, 
            configs["TrainingConfigs"]["n_epoch"], save_folder, target_name, device, 
            configs["Dataset"]["Dataset_Name"], scenario, writer)

    else:
        print(f'{configs["AdaptationConfig"]["Method"]} has not been implemented')
        raise NotImplementedError