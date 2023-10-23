import argparse
import os

import logging
from ml_collections import config_dict
# from utils.model_testing import  Acc_matrix
import wandb
from train.scenario_trainer import DomainTrainer
from train.scenario_evaluator import DomainEvaluator

SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# wandb.tensorboard.patch(root_logdir=os.path.join(os.getcwd(), "logs"))


class FDSweep(DomainTrainer):
    def __init__(self, args):
        super(FDSweep, self).__init__(args)

        """Initialize sweep_params"""
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

    def sweep(self):
        """Run a sweep to find the best hyperparameters."""
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.configs.Method + '_' + self.configs.Dataset_Name,
            "parameters": {**self.sweep_parameters}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)
        wandb.agent(sweep_id, self.train, count=self.num_sweeps)
    def train(self):
        """Handle all scenarios for training and adaptation."""
        # writer_path = os.path.join(os.getcwd(), f"logs/{dataset}/{method}/{scenario}/{target_id}")
        run = wandb.init(config=self.configs.__dict__['_fields'], sync_tensorboard=True)
        self.configs = config_dict.ConfigDict(wandb.config)
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

            overall_report = {metric: round(self.loss_avg_meters[metric].avg, 4) for metric in
                              self.loss_avg_meters.keys()}

            wandb.log(overall_report)
        run.finish()




def parse_arguments():
    """Parse command-line arguments."""
    # ========= Select the DATASET ==============
    parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
    parser.add_argument("--dataset", default="PU_Real", help="Name of the dataset.")
    parser.add_argument('--start-domain', default=0, type=int, help='Manual domain start.')
    # ========= Select the algoritm ==============
    parser.add_argument("--algo", default="SHOT", help="Algorithm to use.")
    # ========= Experiment settings ===============
    parser.add_argument("--writer", default="tensorboard", choices=["tensorboard", "wandb"], help="Logging tool to use.")
    parser.add_argument('-lp', '--log-path', default="./logs", help="Path for logging.")
    parser.add_argument('-e', '--entity', default="", help="Entity for logging.")
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--plot', action='store_true', help="Flag to enable plotting.")
    parser.add_argument('--save', action='store_true', help="Flag to enable saving.")
    # ======== sweep settings =====================
    parser.add_argument("--sweep", action='store_true', help="Flag to enable sweep.")
    parser.add_argument('--num_sweeps', default=50, type=str, help='Number of sweep runs')
    parser.add_argument('--sweep_project_wandb', default='CDA_FD', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', default='timeseries-cda', type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--hp_search_strategy', default="random", type=str,
                        help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_minimize', default="avg_loss", type=str,
                        help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    diagnostic = FDSweep(args)
    diagnostic.sweep()
