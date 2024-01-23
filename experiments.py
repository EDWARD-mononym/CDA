import argparse
import importlib
import os
import logging
import torch
import json
from ml_collections import config_dict

from utils.set_seed import set_seed
from train.scenario_trainer import DomainTrainer
from train.scenario_evaluator import DomainEvaluator

torch.backends.cudnn.deterministic = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FDTrain(DomainTrainer):
    def __init__(self, args):
        super(FDTrain, self).__init__(args)

    def load_configs(self):
        """Load default configuration based on dataset."""
        with open(f"configs/experiments/{self.args.dataset}.json", 'r') as f:
            all_configs = json.load(f)
        general_configs = all_configs['train_params']
        algo_configs = all_configs['algo_params'].get(self.args.algo, None)
        default_configs = config_dict.ConfigDict({**general_configs, **algo_configs})

        """Load sweep configuration based on dataset."""
        with open(f"sweep_configs/experiments/{self.args.dataset}.json", 'r') as f:
            all_configs = json.load(f)
        sweep_algo_configs = all_configs['algo_params'].get(self.args.algo, None)
        return default_configs, sweep_algo_configs

    def load_algorithm(self, configs):
        """Load the specified algorithm."""
        algo_module = importlib.import_module(f"algorithms.experiments.{self.args.experiment}.{self.args.algo}")
        algo_class = getattr(algo_module, self.args.algo)
        return algo_class(configs)

    def save_run_results(self, scenario, run):
        """Save the results after training and adaptation."""
        self.evaluator.calc_metric()
        file_name = os.path.join(os.getcwd(), f'experiment_results/{self.configs.Dataset_Name}/{self.args.experiment}/{self.args.algo}/{scenario}/Run_{run}')
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        self.evaluator.save_singlerun(file_name)

    def list_algos(self):
        path = os.path.join(os.getcwd(), "algorithms", "experiments", self.args.experiment)
        return [f[:-3] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.py')]

    def train(self):
        #? Get all the algorithm
        algolist = self.list_algos()
        for algo in algolist:
            self.args.algo = algo
            """Handle all scenarios for training and adaptation."""
            for scenario in self.configs.Scenarios:
                source_name = scenario[0]

                # Initialize evaluator matrix
                self.evaluator = DomainEvaluator(self.device, scenario, self.configs)

                print("===============================================")
                print("                   CONFIG INFO                  ")
                print("===============================================")
                print(f"Method: {self.configs.Method}")
                print(f"Dataset: {self.configs.Dataset_Name}")
                print(f"Scenario: {' → '.join(scenario)}")
                print("===============================================")

                for run in range(args.Nruns):
                    set_seed(run)

                    # Train source model and log performance
                    self.train_and_log_source_model(source_name, scenario)

                    # Adapt to all target domains
                    self.adapt_to_target_domains(scenario)

                    # Calculate metrics and save results
                    self.save_run_results(scenario, run)

                self.save_avg_runs(scenario)


def parse_arguments():
    """Parse command-line arguments."""
    # ========= Select the DATASET ==============
    parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
    parser.add_argument('--Nruns', default=5, type=int, help='Number of runs')
    parser.add_argument("--dataset", default="PU_Artificial")
    parser.add_argument('--start-domain', default=0, type=int, help='Manual domain start.')
    # ======== Select the experiment =============
    parser.add_argument("--experiment", default="Ablation", help="Experiments available: Ablation, Replay, Stability")
    # ========= Experiment settings ===============
    parser.add_argument("--writer", default="tensorboard", choices=["tensorboard", "wandb"], help="Logging tool to use.")
    parser.add_argument('-lp', '--log-path', default="./logs", help="Path for logging.")
    parser.add_argument('-e', '--entity', default="", help="Entity for logging.")
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--plot', action='store_true', help="Flag to enable plotting.")
    parser.add_argument('--save', action='store_true', help="Flag to enable saving.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.experiment == "Stability":
        args.Nruns = 10
    diagnostic = FDTrain(args)
    diagnostic.train()
