import importlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from configs import Config
from utils import set_seed

class Experiment():
    def __init__(self, algo_name, dataset_name):
        #* Retrieve the configs of the dataset & hparams and combine them into one config class
        # Dataset configs
        dataset_module = importlib.import_module(f"Dataset_configs.{dataset_name}")
        dataset_config_class = getattr(dataset_module, dataset_name)
        dataset_configs = dataset_config_class()
        # Hparams
        hparam_moudle = importlib.import_module(f"Dataset_hparams.{dataset_name}")
        hparam_class = getattr(hparam_moudle, dataset_name)
        hparams = hparam_class()
        # Combine dataset & Hparam config
        self.configs = Config(dataset_configs, hparams, algo_name)

        #* Create folder path to log the algorithm's performance
        self.log_path = os.path.join(os.getcwd(), "logs", dataset_name, algo_name)
        if not os.path.exists(self.log_path): 
            os.makedirs(self.log_path)

        self.model_path = os.path.join(os.getcwd(), "models", dataset_name, algo_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        #* Import the algorithm class
        algo_module = importlib.import_module(f"algorithms.{algo_name}")
        self.algorithm_class = getattr(algo_module, algo_name)

        #* Save the algo name and dataset name for future use
        self.algo_name = algo_name
        self.dataset_name = dataset_name


    def main(self, plot=False):
        for scenario in self.configs.scenarios:
            #? Scenario takes the form (S, T1, T2, ...)
            for run_id in range(self.configs.num_runs):
                set_seed(run_id)

                #* Initialise the algorithm class
                self.algorithm = self.algorithm_class(self.configs)
                self.algorithm.to(self.configs.device)

                #* Initialise the result table
                #! Modify so we can take average of multiple runs
                self.result_table = pd.DataFrame(columns=list(scenario))

                #* Create a save folder
                self.save_path = os.path.join(self.model_path, str(scenario))
                if not os.path.exists(self.save_path): 
                    os.makedirs(self.save_path)

                #* Load source training data
                src_loader = self.get_loader(scenario[0], "train")
                test_loader = self.get_loader(scenario[1], "test")

                #* Train source model
                if plot:
                    self.algorithm_train_source(self.algorithm, src_loader, scenario[0], self.save_path, test_loader)
                else:
                    self.algorithm_train_source_plotless(self.algorithm, src_loader, scenario[0], self.save_path)

                #* Performance of source model & Save
                self.evaluate_model(scenario, scenario[0], run_id)

                for target_id in scenario[1:]:
                    trg_loader = self.get_loader(target_id, "train")                 

                    #* Adapt model to target
                    if plot:
                        self.algorithm_update(self.algorithm, src_loader, trg_loader, target_id, self.save_path, test_loader)
                    else:
                        self.algorithm_update_plotless(self.algorithm, src_loader, trg_loader, target_id, self.save_path)

                    #* Measure performance of current model & Save
                    self.evaluate_model(scenario, target_id, run_id)

                #* Evaluate metrics & Save
                self.evaluate_metric(scenario, run_id)

                if plot:
                    self.algorithm.epoch_performance['Average acc'].plot(marker='.', linestyle=':', color='blue')
                    plt.title(f"{scenario} performance. Epoch: {self.configs.train_params['N_epochs']}")
                    plt.show()

    def get_loader(self, domain_id, dtype): #* Gets the data loader of a specific domain
        path_to_loader = os.path.join(os.getcwd(), f"dataloader\\{self.dataset_name}\\{domain_id}.py")

        spec = importlib.util.spec_from_file_location("loader", path_to_loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if dtype == "train":
            loader = module.trainloader
        elif dtype == "test":
            loader = module.testloader
        else:
            raise NotImplementedError

        return loader

    def algorithm_update_plotless(self, algo_class, src_loader, trg_loader, target_id, save_path):
        algo_class.update(src_loader, trg_loader, target_id, save_path)

    def algorithm_update(self, algo_class, src_loader, trg_loader, target_id, save_path, test_loader):
        algo_class.update(src_loader, trg_loader, target_id, save_path, test_loader=test_loader)

    def algorithm_train_source_plotless(self, algo_class, src_loader, src_id, save_path):
        algo_class.train_source(src_loader, src_id, save_path)

    def algorithm_train_source(self, algo_class, src_loader, src_id, save_path, test_loader):
        algo_class.train_source(src_loader, src_id, save_path, test_loader=test_loader)

    def evaluate_model (self, scenario, current_domain, run_id, best=False):
        #? Measure the performance of the trained/adapted model
        #? Called right after the model has finish training/adapting to a source/target model
        results_row = []

        for test_id in scenario:
            #? Scenario takes the form (S, T1, T2, ...)         

            if best: #* Use the best model
                #* Initialise a separate network
                algorithm = self.algorithm_class(self.configs) #! Change this line
                algorithm.to(self.configs.device)

                #* Define separate feature extractor & classifier
                feature_extractor = algorithm.feature_extractor
                classifier = algorithm.classifier

                #* Send to CPU/GPU
                feature_extractor.to(self.configs.device)
                classifier.to(self.configs.device)

                #* Load the saved model
                feature_extractor.load_state_dict(torch.load(os.path.join(self.save_path, f"feature_extractor_{current_domain}.pt")))
                classifier.load_state_dict(torch.load(os.path.join(self.save_path, f"classifier_{current_domain}.pt")))
            
            else: #* Use the last model
                #* Define separate feature extractor & classifier
                feature_extractor = self.algorithm.feature_extractor
                classifier = self.algorithm.classifier

                #* Send to CPU/GPU
                feature_extractor.to(self.configs.device)
                classifier.to(self.configs.device)

            #* Eval mode
            feature_extractor.eval()
            classifier.eval()

            #* Evaluate the model
            test_loader = self.get_loader(test_id, "test")   
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

        result_df = pd.DataFrame([results_row], columns=self.result_table.columns, index = [current_domain])
        self.result_table = pd.concat([self.result_table, result_df])

        #* Save the results
        self.result_table.to_csv(os.path.join(self.log_path, f"{scenario}_{run_id}.csv"), index = True)
        print(self.result_table)

    def evaluate_metric (self, scenario, run_id):
        #? Measure the ACC, BWT, Adaptability and Transferability for each timestep
        #? Function is called after the performance of all model has been evaluated

        ACC, BWT, Adaptability = [], [], []

        for timestep, domain_id in enumerate(scenario):
            #? Scenario takes the form (S, T1, T2, ...)

            #* ACC
            ACC_list = []
            for i in range(timestep+1):
                ACC_list.append(self.result_table.loc[scenario[timestep], scenario[i]])
            ACC.append(sum(ACC_list) / len(ACC_list))

            #* BWT
            BWT_list = []
            for i in range(timestep):
                BWT_list.append(self.result_table.loc[scenario[timestep], scenario[i]] - self.result_table.loc[scenario[i], scenario[i]])
            try:
                BWT.append(sum(BWT_list) / len(BWT_list))
            except ZeroDivisionError:
                BWT.append(0)

            #* Adaptability
            Adaptability_list = []
            for i in range(1, timestep+1):
                Adaptability_list.append(self.result_table.loc[scenario[i], scenario[i]] - self.result_table.loc[scenario[i-1], scenario[i]])
            try:
                Adaptability.append(sum(Adaptability_list) / len(Adaptability_list))
            except ZeroDivisionError:
                Adaptability.append(0)


        metric_df = pd.DataFrame({"ACC": ACC,
                                  "BWT": BWT,
                                  "Adaptability": Adaptability},
                                  index=scenario)
        self.result_table = pd.concat([self.result_table, metric_df], axis=1)

        #* Save the results
        self.result_table.to_csv(os.path.join(self.log_path, f"{scenario}_{run_id}.csv"), index = True)
        print(self.result_table)
        

if __name__ == "__main__":
    experiment = Experiment("DANN", "DomainNet")
    experiment.main(plot=False)