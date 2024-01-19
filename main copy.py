import argparse
import os
import logging
import torch

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

    def train(self):
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
    parser.add_argument("--dataset", default="PU_Artificial", help="Name of the dataset.")
    parser.add_argument('--start-domain', default=0, type=int, help='Manual domain start.')
    # ========= Select the algoritm ==============
    parser.add_argument("--algo", default="EverAdapt_test", help="Algorithm to use: DeepCORAL, DANN, ")
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
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alpha_values:
        args.alpha = alpha
        diagnostic = FDTrain(args)
        diagnostic.train()
