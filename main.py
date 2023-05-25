import numpy as np
import pandas as pd
import torch
import importlib
import os


from dataloader import create_dataloader
from configs import Config
from utils import set_seed

class Experiment():
    def __init__(self, algo_name, dataset_name):
        #* Retrieving the configs of the dataset & hparams and combining them into one config class
        dataset_module = importlib.import_module(f"Dataset_configs.{dataset_name}")
        dataset_config_class = getattr(dataset_module, dataset_name)
        dataset_configs = dataset_config_class()

        hparam_moudle = importlib.import_module(f"Dataset_hparams.{dataset_name}")
        hparam_class = getattr(hparam_moudle, dataset_name)
        hparams = hparam_class()

        self.configs = Config(dataset_configs, hparams, algo_name)

        #* Setting up the algorithm & file paths
        self.dataset_path = os.path.join(os.getcwd(), f"Data/{dataset_name}")
        self.log_path = os.path.join(os.getcwd(), "logs", dataset_name, algo_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        algo_module = importlib.import_module(f"algorithms.{algo_name}")
        self.algorithm_class = getattr(algo_module, algo_name)


    def main(self):
        
        for run_id in range(self.configs.num_runs):
            set_seed(run_id)

            for scenario in self.configs.scenarios:
                self.algorithm = self.algorithm_class(self.configs)
                self.algorithm.to(self.configs.device)

                #? Scenario takes the form (S, T1, T2, ...)

                #* Define result table
                result_table = pd.DataFrame(columns=list(scenario))

                #* Load source training data
                src_loader = create_dataloader(self.dataset_path, scenario[0], self.configs, "train")

                #* Train source model
                self.algorithm.train_source(src_loader)
                #* Performance of source model
                result_row = self.evaluate_scenario(scenario)
                result_df = pd.DataFrame([result_row], columns=result_table.columns)
                result_table = pd.concat([result_table, result_df], ignore_index=True)

                for timestep, target_id in enumerate(scenario[1:]):
                    trg_loader = create_dataloader(self.dataset_path, target_id, self.configs, "train")

                    self.algorithm.update(src_loader, trg_loader, timestep+1)

                    #* Measure performance of current model
                    result_row = self.evaluate_scenario(scenario)
                    result_df = pd.DataFrame([result_row], columns=result_table.columns)
                    result_table = pd.concat([result_table, result_df], ignore_index=True)
                    # print(result_table)

                #* Save the results
                result_table.to_csv(os.path.join(self.log_path, f"{scenario}_{run_id}.csv"), index = True)
                print(result_table)

    def evaluate_scenario (self, scenario):
        
        results_row = []

        for test_id in scenario:
            #? Scenario takes the form (S, T1, T2, ...)
            test_loader = create_dataloader(self.dataset_path, test_id, self.configs, "test")

            feature_extractor = self.algorithm.feature_extractor.to(self.configs.device)
            classifier = self.algorithm.classifier.to(self.configs.device)

            #* Eval mode
            feature_extractor.eval()
            classifier.eval()

            acc_list = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.float().to(self.configs.device)
                    y = y.view((-1)).long().to(self.configs.device)

                    #* Forward pass
                    features = feature_extractor(x)
                    predict_logits = classifier(features)
                    pred = predict_logits.argmax(dim=1)

                    #* Compute accuracy
                    acc = torch.eq(pred, y).sum().item() / len(y)
                    acc_list.append(acc)

            results_row.append(torch.tensor(acc_list).mean().item())  #* Average accuracy

        return results_row
        

if __name__ == "__main__":
    experiment = Experiment("DANN", "HHAR")
    experiment.main()