import argparse
import importlib
import numpy as np
import os
import random
import time
import torch

from train.adaptation import adapt
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_best_model
from utils.model_testing import test_all_domain, Acc_matrix

# Default settings
parser = argparse.ArgumentParser(description='DA for Fault Diagnostic')
# Dataset Parameters
parser.add_argument("--dataset", default="HHAR")
parser.add_argument("--algo", default="DeepCORAL")
parser.add_argument("--writer", default="tensorboard", help="tensorboard or wandb")
parser.add_argument('-lp', '--log-path', default="./logs")  # log path
parser.add_argument('-e', '--entity', default="")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Info Parameters
parser.add_argument('--start-domain', default=0, type=int, metavar='N',
                    help='manual domain start (useful on restarts)')
args = parser.parse_args()

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def main(args = args):
    #* Load configs
    config_module = importlib.import_module(f"configs.{args.dataset}.{args.algo}")
    configs = getattr(config_module, 'configs')

    for scenario in configs["Dataset"]["Scenarios"]:
        feature_extractor, classifier = load_source_model(configs, scenario, device)

        #* Test source model acc & save
        result_matrix = Acc_matrix(scenario)
        source_accs = test_all_domain(configs["Dataset"]["Dataset_Name"], scenario, feature_extractor, classifier, device)
        result_matrix.update(scenario[0], source_accs)

        for target_id in scenario[1:]: # First index of scenario is assumed to be source domain
            #* Adaptation
            writer = create_writer(configs["Dataset"]["Dataset_Name"], configs["AdaptationConfig"]["Method"], scenario, target_id)
            adapt(feature_extractor, classifier, target_id, scenario, configs, device, writer)
            writer.close()

            #* Load the best model & test acc
            feature_extractor, classifier = load_best_model(configs, scenario, target_id, device)
            target_accs = test_all_domain(configs["Dataset"]["Dataset_Name"], scenario, feature_extractor, classifier, device)
            result_matrix.update(target_id, target_accs)

        #* Calculate Acc, BWT, Adapt and Generalise then save the results
        result_matrix.calc_metric()
        save_folder = os.path.join(os.getcwd(), f'results/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_name = os.path.join(save_folder, f"{scenario}.csv")
        result_matrix.save(file_name)

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    start = time.time()
    main(args)
    end = time.time()
    time_taken = end-start
    print(f"time taken: {time_taken: .2f} seconds")