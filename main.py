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
import json
SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaultDiagnostic:
    def __init__(self, args):
        """Initialize the FaultDiagnostic class."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.setup_seed(SEED)
        self.configs= self.load_configs()
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


        return default_configs

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

    def handle_scenarios(self):
        """Handle all scenarios for training and adaptation."""
        for scenario in self.configs.Scenarios:
            source_name = scenario[0]

            # Initialize accuracy matrix
            self.result_matrix = Acc_matrix(scenario)

            # Train source model and log performance
            self.train_and_log_source_model(source_name, scenario)

            # Adapt to all target domains
            self.adapt_to_target_domains(scenario)

            # Calculate metrics and save results
            self.save_results(scenario)

        overall_report = {metric: round(self.loss_avg_meters[metric].avg, 2) for metric in
                          self.loss_avg_meters.keys()}
        logging.debug(overall_report)
    def run(self):
        """Main run method for the FaultDiagnostic class."""
        logging.info("Starting the FaultDiagnostic run.")
        self.handle_scenarios()
        logging.info("Completed the FaultDiagnostic run.")

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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
    parser.add_argument("--dataset", default="PU_Real", help="Name of the dataset.")
    parser.add_argument("--algo", default="DeepCORAL", help="Algorithm to use.")
    parser.add_argument("--writer", default="tensorboard", choices=["tensorboard", "wandb"], help="Logging tool to use.")
    parser.add_argument('-lp', '--log-path', default="./logs", help="Path for logging.")
    parser.add_argument('-e', '--entity', default="", help="Entity for logging.")
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--plot', action='store_true', help="Flag to enable plotting.")
    parser.add_argument('--save', action='store_true', help="Flag to enable saving.")
    parser.add_argument('--start-domain', default=0, type=int, help='Manual domain start.')
    parser.add_argument("--sweep", action='store_true', help="Flag to enable sweep.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    diagnostic = FaultDiagnostic(args)
    diagnostic.run()
