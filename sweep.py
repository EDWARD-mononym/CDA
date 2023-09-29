from collections import defaultdict
import importlib
from itertools import product
import os

from train.adaptation import adapt
from train.pretrain import pretrain
from utils.avg_meter import AverageMeter
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_best_model
from utils.model_testing import test_all_domain, Acc_matrix
from utils.save_sweep import dicts_to_csv

def sweep(args, device):
    #* Load configs
    config_module = importlib.import_module(f"sweep_configs.{args.dataset}.{args.algo}")
    configs = getattr(config_module, 'configs')

    hyperparameters = configs["Hyperparameters"]
    hyperparameter_combinations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

    for hyperparameter_selection in hyperparameter_combinations:

        #* Load algorithm class
        algo_module = importlib.import_module(f"algorithms.{args.algo}")
        algo_class = getattr(algo_module, args.algo)
        algo = algo_class(configs, hyperparameter_selection)

        #* Create loss meter
        loss_avg_meters = defaultdict(lambda: AverageMeter())

        for scenario in configs["Dataset"]["Scenarios"]:
            source_name = scenario[0]

            #* Initialise R matrix (acc matrix)
            result_matrix = Acc_matrix(scenario)

            #* Train source model & log source performance of best model
            pretrain(algo, source_name=source_name, configs=configs, device=device)
            algo.feature_extractor, algo.classifier = load_source_model(configs, algo.feature_extractor, algo.classifier,
                                                                        scenario, device)
            source_accs = test_all_domain(configs["Dataset"]["Dataset_Name"], scenario,
                                        algo.feature_extractor, algo.classifier, device)
            result_matrix.update(source_name, source_accs)

            #* Adapt to all target domains
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

            #* Calculate Acc, BWT, Adapt and Generalise and log into loss_avg_meter
            result_matrix.calc_metric()

            loss_avg_meters["avg_acc"].update(result_matrix.acc.iloc[1:]['ACC'].mean())
            loss_avg_meters["avg_bwt"].update(result_matrix.bwt.iloc[2:]['BWT'].mean())
            loss_avg_meters["avg_adapt"].update(result_matrix.adapt.iloc[1:]["Adapt"].mean())
            loss_avg_meters["avg_generalise"].update(result_matrix.generalise.iloc[1:-1]["Generalise"].mean())

        #* Save results into csv with Hparameter values
        save_folder = os.path.join(os.getcwd(), f'sweep_results/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}')
        save_file = os.path.join(save_folder, f"sweep_result.csv")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            # Save with header
            dicts_to_csv(hyperparameter_selection, loss_avg_meters, save_file, headings=True)

        else:
            dicts_to_csv(hyperparameter_selection, loss_avg_meters, save_file, headings=False)