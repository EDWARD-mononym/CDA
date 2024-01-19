import importlib
import os
import random
import torch
import numpy as np
from collections import defaultdict
import logging
from ml_collections import config_dict
from utils.avg_meter import AverageMeter
import json
SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
from pathlib import Path

from utils.get_loaders import get_loader
# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Abstract_train:
    def __init__(self, args):
        """Initialize the FaultDiagnostic class."""
        self.evaluator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.setup_seed(SEED)
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

    def calc_overall_metrics(self): #! NeedToChange
        self.loss_avg_meters["avg_acc"].update(self.evaluator.acc.iloc[1:]['ACC'].mean())
        self.loss_avg_meters["avg_bwt"].update(self.evaluator.bwt.iloc[2:]['BWT'].mean())
        self.loss_avg_meters["avg_adapt"].update(self.evaluator.adapt.iloc[1:]["Adapt"].mean())
        # self.loss_avg_meters["avg_generalise"].update(self.evaluator.generalise.iloc[1:-1]["Generalise"].mean())

    def save_run_results(self, scenario, run):
        """Save the results after training and adaptation."""
        self.evaluator.calc_metric()
        file_name = os.path.join(os.getcwd(), f'results/{self.configs.Dataset_Name}/{self.args.algo}/{scenario}/Run_{run}')
        # file_name = os.path.join(os.getcwd(), f'results/{self.configs.Dataset_Name}/{self.args.algo}/{str(self.args.alpha)}/{scenario}/Run_{run}') #! MODIFIED
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        self.evaluator.save_singlerun(file_name)

    def save_avg_runs(self, scenario):
        save_folder = os.path.join(os.getcwd(), f'results/{self.configs.Dataset_Name}/{self.args.algo}/{scenario}')
        # save_folder = os.path.join(os.getcwd(), f'results/{self.configs.Dataset_Name}/{self.args.algo}/{str(self.args.alpha)}/{scenario}') #! MODIFIED
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.evaluator.save_overall(save_folder)

        # with open(os.path.join(save_folder, "configs_used.json"), 'w') as file:
        #     json.dump(self.configs, file)

    def pretrain(self, source_name):
        train_loader = get_loader(self.configs.Dataset_Name, source_name, "train")
        test_loader = get_loader(self.configs.Dataset_Name, source_name, "test")

        save_folder = os.path.join(os.getcwd(), f"source_models/{self.configs.Dataset_Name}/{self.configs.Backbone_Type}")
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        self.algo.pretrain(train_loader, test_loader, source_name, save_folder, self.device, self.evaluator)

    def adapt(self, target_name, scenario, writer):
        trg_loader = get_loader(self.configs.Dataset_Name, target_name, "train")
        src_loader = get_loader(self.configs.Dataset_Name, scenario[0], "train")

        save_path = os.path.join(os.getcwd(), f'adapted_models/{self.configs.Dataset_Name}/{self.configs.Method}/{scenario}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.algo.update(src_loader, trg_loader,
                          scenario, target_name, self.configs.Dataset_Name,
                          save_path, writer, self.device, self.evaluator, self.loss_avg_meters)


    def test_domain(self, test_loader):
        self.algo.feature_extractor.eval()
        self.algo.classifier.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0], data[1]
                x, y = x.to(self.device), y.to(self.device)
                logits = self.algo.classifier(self.algo.feature_extractor(x))
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        accuracy = correct / total
        return accuracy

    def test_all_domain(self, scenario):
        acc_dict = defaultdict(float)
        for domain in scenario:
            test_loader = get_loader(self.configs.Dataset_Name, domain, "test")
            acc = self.test_domain(test_loader)
            acc_dict[domain] = acc
        return acc_dict