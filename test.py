import random
import numpy as np
import torch

from dataloader import load_scenarios

from algorithms.BaseModel import BaseModel
from algorithms.DeepCoral import DeepCoral
from algorithms.DANN import DANN
from algorithms.IdealModel import IdealModel

from constants import CONFIG

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

class Experiment():
    def __init__(self, configs):
        self.configs = configs

        #* Load datasets
        self.train_loaders, self.test_loaders = load_scenarios(self.configs.datapath, self.configs.scenario)

        #* Prepare models
        SourceModelConfigs = self.configs
        SourceModelConfigs.saved_models_path = "C:/Work/ASTAR/codes/CDA/CDA/Trained_models/Source"
        self.SourceModel = BaseModel(SourceModelConfigs)
        self.SourceModel.to(self.configs.device)

        DeepCoralConfigs = self.configs
        DeepCoralConfigs.saved_models_path = "C:/Work/ASTAR/codes/CDA/CDA/Trained_models/DeepCoral"
        self.DeepCoral = DeepCoral(DeepCoralConfigs)
        self.DeepCoral.to(self.configs.device)

        DANNConfigs = self.configs
        DANNConfigs.saved_models_path = "C:/Work/ASTAR/codes/CDA/CDA/Trained_models/DANN"
        self.DANN = DANN(DANNConfigs)
        self.DANN.to(self.configs.device)

        IdealConfigs = self.configs
        IdealConfigs.saved_models_path = "C:/Work/ASTAR/codes/CDA/CDA/Trained_models/Ideal"
        self.IdealModel = IdealModel(IdealConfigs)
        self.IdealModel.to(self.configs.device)

    def train(self):
        #* Initialise the source models
        self.SourceModel.train_source(self.train_loaders)
        # self.DeepCoral.train_source(self.train_loaders)
        # self.DANN.train_source(self.train_loaders)

        # #* Train a model for each domain with UDA
        # for timestep, scenario in enumerate(self.configs.scenario[1:]):
        #     self.DeepCoral.update(self.train_loaders, timestep+1)
        #     self.DANN.update(self.train_loaders, timestep+1)

        # #* Train an ideal model
        # self.IdealModel.train(self.train_loaders)

if __name__ == "__main__":
    test = Experiment(CONFIG)
    test.train()