import argparse
import importlib
import os
import random
import time
import torch
import numpy as np
from collections import defaultdict
import logging
from ml_collections import config_dict
from utils.get_loaders import get_loader
from train.pretrain import pretrain
from utils.avg_meter import AverageMeter
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_best_model
from utils.model_testing import test_all_domain, Acc_matrix
import wandb
import json
SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Abstract_sweep:
    def __init__(self, args):
        """Initialize the FaultDiagnostic class."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.setup_seed(SEED)
        self.configs, self.sweep_paramters = self.load_configs()
        self.algo = self.load_algorithm(self.configs)

        self.loss_avg_meters = defaultdict(lambda: AverageMeter())

    @staticmethod
    def setup_seed(seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load_configs(self):
        """Load default configuration based on dataset."""
        with open(f"configs/{self.args.dataset}.json", 'r') as f:
            all_configs = json.load(f)
        general_configs = all_configs['train_params']
        algo_configs = all_configs['algo_params'].get(self.args.algo, None)
        default_configs = config_dict.ConfigDict({**general_configs, **algo_configs})

        """Load sweep configuration based on dataset."""
        with open(f"sweep_configs/{self.args.dataset}.json", 'r') as f:
            all_configs = json.load(f)
        sweep_algo_configs = all_configs['algo_params'].get(self.args.algo, None)
        return default_configs, sweep_algo_configs

    def load_algorithm(self, configs):
        """Load the specified algorithm."""
        algo_module = importlib.import_module(f"algorithms.{self.args.algo}")
        algo_class = getattr(algo_module, self.args.algo)
        return algo_class(configs)

    def train_and_log_source_model(self, source_name, scenario):
        """Train the source model and log its performance."""
        pretrain(self.algo, source_name=source_name, configs=self.configs, device=self.device)
        self.algo.feature_extractor, self.algo.classifier = load_source_model(self.configs,
                                                                              self.algo.feature_extractor,
                                                                              self.algo.classifier, scenario,
                                                                              self.device)
        source_accs = test_all_domain(self.configs.Dataset_Name, scenario, self.algo.feature_extractor,
                                      self.algo.classifier, self.device)
        self.result_matrix.update(source_name, source_accs)

    def adapt_to_target_domains(self, scenario):
        """Adapt the model to all target domains."""
        for target_name in scenario[1:]:
            writer = create_writer(self.configs.Dataset_Name, self.args.algo, scenario, target_name)
            trg_loader = get_loader(self.configs.Dataset_Name, target_name, "train")
            src_loader = get_loader(self.configs.Dataset_Name, scenario[0], "train")

            save_path = os.path.join(os.getcwd(), f'adapted_models/{self.configs.Dataset_Name}/{self.configs.Method}/{scenario}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            self.algo.update(src_loader, trg_loader, scenario, target_name, self.configs.Dataset_Name, save_path,
                             writer, self.device, self.loss_avg_meters)
            writer.close()

            # Load the best model and test accuracy
            self.algo.feature_extractor, self.algo.classifier = load_best_model(self.configs,
                                                                                self.algo.feature_extractor,
                                                                                self.algo.classifier, scenario,
                                                                                target_name, self.args.algo,
                                                                                self.device)
            target_accs = test_all_domain(self.configs.Dataset_Name, scenario, self.algo.feature_extractor,
                                          self.algo.classifier, self.device)
            self.result_matrix.update(target_name, target_accs)

        self.result_matrix.calc_metric()
        self.loss_avg_meters["avg_acc"].update(self.result_matrix.acc.iloc[1:]['ACC'].mean())
        self.loss_avg_meters["avg_bwt"].update(self.result_matrix.bwt.iloc[2:]['BWT'].mean())
        self.loss_avg_meters["avg_adapt"].update(self.result_matrix.adapt.iloc[1:]["Adapt"].mean())
        self.loss_avg_meters["avg_generalise"].update(self.result_matrix.generalise.iloc[1:-1]["Generalise"].mean())
    def save_results(self, scenario):
        """Save the results after training and adaptation."""
        self.result_matrix.calc_metric()
        save_folder = os.path.join(os.getcwd(), f'results/{self.configs.Dataset_Name}/{self.args.algo}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        folder_name = os.path.join(save_folder, f"{scenario}")
        self.result_matrix.save(folder_name)
        if self.args.plot:
            plot_file = os.path.join(save_folder, f"{scenario}.png")
            self.result_matrix.save_plot(plot_file)
