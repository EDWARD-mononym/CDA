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
from train.sweep import Abstract_sweep
SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaultDiagnostic(Abstract_sweep):
    def __init__(self, args):
        super(FaultDiagnostic, self).__init__(args)

        """Initialize sweep_params"""

    def sweep(self):
        """Run a sweep to find the best hyperparameters."""
        sweep_runs_count = 10
        sweep_config = {
            "method": "random",
            "metric": {"name": "avg_loss", "goal": "minimize"},
            "parameters": {**self.sweep_paramters}
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="test_project")
        wandb.agent(sweep_id, self.train, count=sweep_runs_count)
    def train(self):
        """Handle all scenarios for training and adaptation."""
        run = wandb.init(config=self.configs.__dict__['_fields'])
        self.configs = config_dict.ConfigDict(wandb.config)
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

        wandb.log(overall_report)
        run.finish()




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
    diagnostic.sweep()
