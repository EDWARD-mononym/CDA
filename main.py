import argparse
import os
import logging

from train.scenario_trainer import DomainTrainer
from train.scenario_evaluator import DomainEvaluator
SEED = 42
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
            self.evaluator = DomainEvaluator(self.algo, self.device, scenario, self.configs)

            # Train source model and log performance
            self.train_and_log_source_model(source_name, scenario)

            # Adapt to all target domains
            self.adapt_to_target_domains(scenario)

            # Calculate metrics and save results
            self.save_results(scenario)


def parse_arguments():
    """Parse command-line arguments."""
    # ========= Select the DATASET ==============
    parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
    parser.add_argument("--dataset", default="PU_Real", help="Name of the dataset.")
    parser.add_argument('--start-domain', default=0, type=int, help='Manual domain start.')
    # ========= Select the algoritm ==============
    parser.add_argument("--algo", default="NRC", help="Algorithm to use: DeepCORAL, DANN, ")
    # ========= Experiment settings ===============
    parser.add_argument("--writer", default="tensorboard", choices=["tensorboard", "wandb"], help="Logging tool to use.")
    parser.add_argument('-lp', '--log-path', default="./logs", help="Path for logging.")
    parser.add_argument('-e', '--entity', default="", help="Entity for logging.")
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--plot', action='store_true', help="Flag to enable plotting.")
    parser.add_argument('--save', action='store_true', help="Flag to enable saving.")
    # ======== sweep settings =====================
    parser.add_argument("--sweep", action='store_true', help="Flag to enable sweep.")
    parser.add_argument('--num_sweeps', default=1, type=str, help='Number of sweep runs')
    parser.add_argument('--sweep_project_wandb', default='Test_CDA', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--hp_search_strategy', default="random", type=str,
                        help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_minimize', default="avg_loss", type=str,
                        help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    diagnostic = FDTrain(args)
    diagnostic.train()
