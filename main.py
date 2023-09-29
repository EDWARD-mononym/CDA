import argparse
from collections import defaultdict
import importlib
import numpy as np
import os
import random
import time
import torch

from sweep import sweep
from train.adaptation import adapt
from train.pretrain import pretrain
from utils.avg_meter import AverageMeter
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_best_model
from utils.model_testing import test_all_domain, Acc_matrix

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Default settings
parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
# Dataset Parameters
parser.add_argument("--dataset", default="PU_Real")
parser.add_argument("--algo", default="DeepCORAL")
parser.add_argument("--writer", default="tensorboard", help="tensorboard or wandb")
parser.add_argument('-lp', '--log-path', default="./logs")  # log path
parser.add_argument('-e', '--entity', default="")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--plot', default=True)
parser.add_argument('--save', default=False)
# Train Info Parameters
parser.add_argument('--start-domain', default=0, type=int, metavar='N',
                    help='manual domain start (useful on restarts)')
# Sweep
parser.add_argument("--sweep", default="True")
args = parser.parse_args()

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


####### Info #######
# ? configs should be a file in configs folder which contain important values of certain parameters
# ? algo should be defined in one of the file in algorithms folder
# ? R_matrix is defined in utils/model_testing

def main(args=args):
    # * Load configs
    config_module = importlib.import_module(f"configs.{args.dataset}.{args.algo}")
    configs = getattr(config_module, 'configs')

    hyperparameters = configs["Hyperparameters"]

    # * Load algorithm class
    algo_module = importlib.import_module(f"algorithms.{args.algo}")
    algo_class = getattr(algo_module, args.algo)
    algo = algo_class(configs, hyperparameters)

    #* Create loss meter
    loss_avg_meters = defaultdict(lambda: AverageMeter())

    for scenario in configs["Dataset"]["Scenarios"]:
        source_name = scenario[0]

        # * Initialise R matrix (acc matrix)
        result_matrix = Acc_matrix(scenario)

        # * Train source model & log source performance of best model
        pretrain(algo, source_name=source_name, configs=configs, device=device)
        algo.feature_extractor, algo.classifier = load_source_model(configs, algo.feature_extractor, algo.classifier,
                                                                    scenario, device)
        source_accs = test_all_domain(configs["Dataset"]["Dataset_Name"], scenario,
                                      algo.feature_extractor, algo.classifier, device)
        result_matrix.update(source_name, source_accs)

        # * Adapt to all target domains
        for target_name in scenario[1:]:  # First index of scenario is assumed to be source domain
            # * Adapt & log training progress on writer
            writer = create_writer(configs["Dataset"]["Dataset_Name"], configs["AdaptationConfig"]["Method"], scenario,
                                   target_name)
            adapt(algo, target_name, scenario, configs, writer, device, loss_avg_meters)
            writer.close()

            # * Load the best model & test acc
            algo.feature_extractor, algo.classifier = load_best_model(configs, algo.feature_extractor, algo.classifier,
                                                                      scenario, target_name, device)
            target_accs = test_all_domain(configs["Dataset"]["Dataset_Name"], scenario,
                                          algo.feature_extractor, algo.classifier, device)
            result_matrix.update(target_name, target_accs)

        # * Calculate Acc, BWT, Adapt and Generalise then save the results
        result_matrix.calc_metric()
        save_folder = os.path.join(os.getcwd(), f'results/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_name = os.path.join(save_folder, f"{scenario}")
        result_matrix.save(file_name)
        if args.plot:
            plot_file = os.path.join(save_folder, f"{scenario}.png")
            result_matrix.save_plot(plot_file)


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    start = time.time()

    if args.sweep:
        sweep(args, device)
    else:
        main(args)

    end = time.time()

    time_taken = end - start
    print(f"time taken: {time_taken: .2f} seconds")