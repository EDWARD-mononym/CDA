import importlib
import os
from pathlib import Path
import torch
from torch.optim.lr_scheduler import StepLR

from utils.get_loaders import get_loader

def adapt(feature_extractor, classifier, target_name, scenario, configs, device, writer):
    #* Load data & define optimisers
    train_trg_loader = get_loader(configs["Dataset"]["Dataset_Name"], target_name, "train")
    test_trg_loader = get_loader(configs["Dataset"]["Dataset_Name"], target_name, "test")
    src_loader = get_loader(configs["Dataset"]["Dataset_Name"], scenario[0], "train")

    feature_extractor_optimiser = torch.optim.Adam(
            feature_extractor.parameters(),
            lr=configs["OptimiserConfig"]["lr"],
            weight_decay=configs["OptimiserConfig"]["weight_decay"]
        )

    classifier_optimiser = torch.optim.Adam(
            feature_extractor.parameters(),
            lr=configs["OptimiserConfig"]["lr"],
            weight_decay=configs["OptimiserConfig"]["weight_decay"]
        )
        
    fe_lr_scheduler = StepLR(feature_extractor_optimiser, step_size=configs["OptimiserConfig"]['step_size'], gamma=configs["OptimiserConfig"]['gamma'])
    classifier_lr_scheduler = StepLR(feature_extractor_optimiser, step_size=configs["OptimiserConfig"]['step_size'], gamma=configs["OptimiserConfig"]['gamma'])

    save_folder = os.path.join(os.getcwd(), f'adapted_models/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}/{scenario}')
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    #* Adaptation
    if configs["AdaptationConfig"]["Method"] == "DeepCORAL":
        imported_algo = importlib.import_module(f"algorithms.DeepCORAL")
        DeepCORAL = getattr(imported_algo, "DeepCORAL") #? These two lines are equivalent to "from algorithms.DeepCORAL import DeepCORAL"
        DeepCORAL(src_loader, train_trg_loader, 
            feature_extractor, classifier, 
            feature_extractor_optimiser, classifier_optimiser, fe_lr_scheduler, classifier_lr_scheduler,
            configs["TrainingConfigs"]["n_epoch"], save_folder, target_name, device, 
            configs["Dataset"]["Dataset_Name"], scenario, writer)

    elif configs["AdaptationConfig"]["Method"] == "DANN":
        imported_algo = importlib.import_module(f"algorithms.DANN")
        DANN = getattr(imported_algo, "DANN")
        Discriminator = getattr(imported_algo, "Discriminator")

        domain_discriminator = Discriminator(configs)
        domain_discriminator_optimiser = torch.optim.Adam(domain_discriminator.parameters(), lr=1e-3)
        DANN(src_loader, train_trg_loader, feature_extractor, classifier, domain_discriminator,
         feature_extractor_optimiser,  classifier_optimiser, domain_discriminator_optimiser,
         configs["TrainingConfigs"]["n_epoch"], save_folder, target_name, device, configs["Dataset"]["Dataset_Name"], scenario, writer)

    else:
        print(f'{configs["AdaptationConfig"]["Method"]} has not been implemented')
        raise NotImplementedError