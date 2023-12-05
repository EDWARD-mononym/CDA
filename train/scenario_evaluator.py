import torch
import torch.nn as nn
import os
import pandas as pd
from utils.avg_meter import AverageMeter
from utils.plot import save_plot
from utils.get_loaders import get_loader
from collections import defaultdict

class DomainEvaluator:
    def __init__(self, device, scenario, configs):
        self.device = device
        self.scenario = scenario
        self.configs = configs
        self.acc_matrix = pd.DataFrame(index=scenario, columns=scenario)

        self.avg_Rmatrix = {(row, col): AverageMeter() for row in self.acc_matrix.index for col in self.acc_matrix.columns}
        self.avg_ACC = AverageMeter()
        self.avg_BWT = AverageMeter()
        self.avg_FWT = AverageMeter()

        self.epoch_acc = {"epoch": [], "domain_name": [], "epoch_acc": []}
        self.avg_epoch_acc = {"epoch": None, "domain_name": None, "epoch_acc": {}}

    def test_domain(self, algo, test_loader):
        algo.to(self.device)
        algo.feature_extractor.eval()
        algo.classifier.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0], data[1]
                x, y = x.to(self.device), y.to(self.device)
                logits = algo.classifier(algo.feature_extractor(x))
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        accuracy = correct / total
        return accuracy

    def test_all_domain(self, algo):
        acc_dict = defaultdict(float)
        for domain in self.scenario:
            test_loader = get_loader(self.configs.Dataset_Name, domain, "test")
            acc = self.test_domain(algo, test_loader)
            acc_dict[domain] = acc
        return acc_dict


    def update(self, model, acc_dict):
        self.acc_matrix[model] = self.acc_matrix.index.map(acc_dict)

    def update_epoch_acc(self, epoch, domain_name, epoch_acc):
        self.epoch_acc["epoch"].append(epoch)
        self.epoch_acc["domain_name"].append(domain_name)
        self.epoch_acc["epoch_acc"].append(epoch_acc)

    def calc_metric(self):  # Calculate ACC, BWT, Adapt, FWT
        self.acc = self.calc_ACC()
        self.bwt = self.calc_BWT()
        self.fwt = self.calc_FWT()

    def calc_ACC(self):
        acc_values = [0.0]  # ACC not counted for souce model, so we start with 0
        for T, domain in enumerate(self.scenario[1:], start=1):  # ! Keep in mind T starts from 1
            acc_t = (1 / T) * self.acc_matrix[domain].iloc[1:(T + 1)].sum()
            acc_values.append(acc_t)
        acc_column = pd.DataFrame(acc_values, index=self.scenario, columns=['ACC'])
        return acc_column

    def calc_BWT(self):
        bwt_values = [0.0, 0.0]  # BWT does not exist for source model and first target, so we start with two 0
        for T, domain in enumerate(self.scenario[2:], start=2):  # ! Keep in mind T starts from 2
            row_names = [self.scenario[i] for i in range(1, T + 1)]
            bwt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, row] for row in row_names) / (T-1)
            bwt_values.append(bwt_t)
        bwt_column = pd.DataFrame(bwt_values, index=self.scenario, columns=['BWT'])
        return bwt_column

    def calc_FWT(self):
        fwt_values = [0.0]  # Generalise does not exist for source model so we start with 0
        for T, domain in enumerate(self.scenario[1:-1],
                                   start=1):  # ! Keep in mind T starts from 1 and ends before last domain
            unseen_rows = [self.scenario[i] for i in range(T + 1, len(self.scenario))]
            n_unseen = len(unseen_rows)
            fwt_t = sum(
                self.acc_matrix.loc[unseen, domain] - self.acc_matrix.loc[unseen, self.scenario[0]] for unseen in
                unseen_rows) / n_unseen
            fwt_values.append(fwt_t)
        fwt_values.append(0.0)  # Add another 0 for end of domain where Generalise does not exist
        fwt_column = pd.DataFrame(fwt_values, index=self.scenario, columns=['FWT'])
        return fwt_column

    def get_src_risk(self, algo, domain, loss_avg_meters):
        src_loader = get_loader(self.configs.Dataset_Name, domain, "test")

        algo.feature_extractor.eval()
        algo.classifier.eval()

        self.loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels, _ in src_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits = algo.classifier(algo.feature_extractor(inputs))
                batch_loss = self.loss_fn(logits, labels)

                total_loss += batch_loss.item() * labels.size(0)
                total_samples += labels.size(0)

        loss_avg_meters["src_risk"].update(total_loss / total_samples)
        # return total_loss / total_samples

    def save_singlerun(self, folder_name):
        self.update_overall()
        with open(os.path.join(folder_name, "ModelPerformance.csv"), 'w') as f:
            for model in self.acc_matrix.columns:  # Save performance of each model
                # Separate performance of each model to make it easier to process
                column_df = self.acc_matrix[model]
                row_df = pd.DataFrame(column_df).T
                # Append metrics into the row df
                row_df["ACC"] = round(self.acc.loc[model, "ACC"] * 100, 2)
                row_df["BWT"] = round(self.bwt.loc[model, "BWT"] * 100, 2)
                row_df["FWT"] = round(self.fwt.loc[model, "FWT"] * 100, 2)

                # Save model performance
                f.write(f"Model {model} performance")
                row_df.to_csv(f, index=True)
                f.write("\n")

        with open(os.path.join(folder_name, "Rmatrix.csv"), 'w') as f:
            f.write("R matrix")
            # Save R matrix
            R_matrix = self.acc_matrix.applymap(lambda x: round(x * 100, 2))
            R_matrix.to_csv(f, index=True)

        epoch_df = pd.DataFrame(self.epoch_acc)
        epoch_df.to_csv(os.path.join(folder_name, "Unfamiliar.csv"))
        self.epoch_acc = {"epoch": [], "domain_name": [], "epoch_acc": []}

    def update_overall(self):
        #* Update avg R matrix
        for col_id, col in enumerate(self.acc_matrix.columns):
            for row in self.acc_matrix.index:
                self.avg_Rmatrix[(row, col)].update(self.acc_matrix.loc[row, col])

        #* Update avg metrics
        self.avg_ACC.update(self.acc.iloc[-1]['ACC'])
        self.avg_BWT.update(self.bwt.iloc[-1]['BWT'])
        self.avg_FWT.update(self.fwt.iloc[-2]['FWT'])

        #* Update avg unfamiliar period
        if not self.avg_epoch_acc["epoch"]:
            self.avg_epoch_acc["epoch"] = self.epoch_acc["epoch"]
            self.avg_epoch_acc["domain_name"] = self.epoch_acc["domain_name"]
        self.avg_epoch_acc["epoch_acc"][len(self.avg_epoch_acc["epoch_acc"])] = self.epoch_acc["epoch_acc"]

    def save_overall(self, folder_name):
        #* Save overall R_matrix
        overall_df = pd.DataFrame(columns=self.acc_matrix.columns, index=self.acc_matrix.index)
        for (row, col) in self.avg_Rmatrix:
            overall_df.at[row, col] = f"{self.avg_Rmatrix[(row, col)].average()* 100:.2f} +/- {self.avg_Rmatrix[(row, col)].standard_deviation()* 100:.2f}"
        overall_df.to_csv(os.path.join(folder_name, "R_matrix.csv"))

        #* Save overall metric csv & latex.txt
        metrics_df = pd.DataFrame(columns = ['ACC', 'BWT', 'FWT'])
        metrics_df.at[0, 'ACC'] = f"{self.avg_ACC.average()* 100:.2f} +/- {self.avg_ACC.standard_deviation()* 100:.2f}"
        metrics_df.at[0, 'BWT'] = f"{self.avg_BWT.average()* 100:.2f} +/- {self.avg_BWT.standard_deviation()* 100:.2f}"
        metrics_df.at[0, 'FWT'] = f"{self.avg_FWT.average()* 100:.2f} +/- {self.avg_FWT.standard_deviation()* 100:.2f}"
        latex_txt = f"{self.avg_ACC.average()* 100:.2f} $\\pm$ {self.avg_ACC.standard_deviation()* 100:.2f} & {self.avg_BWT.average()* 100:.2f} $\\pm$ {self.avg_BWT.standard_deviation()* 100:.2f} & {self.avg_FWT.average()* 100:.2f} $\\pm$ {self.avg_FWT.standard_deviation()* 100:.2f}"

        metrics_df.to_csv(os.path.join(folder_name, "Metrics.csv"))
        with open(os.path.join(folder_name, "Metrics_Latex.txt"), 'w') as f:
            f.write(latex_txt)

        #* Save overall unfamiliar
        N_runs = len(self.avg_epoch_acc["epoch_acc"])
        N_points = len(self.avg_epoch_acc["epoch_acc"][0])
        avg_epoch = [0] * N_points
        for i in range(N_points):
            sum_at_position = 0
            for key in self.avg_epoch_acc["epoch_acc"]:
                sum_at_position += self.avg_epoch_acc["epoch_acc"][key][i]
            avg_epoch[i] = sum_at_position / N_runs
        self.avg_epoch_acc["epoch_acc"] = avg_epoch
        unfamiliar_df = pd.DataFrame(self.avg_epoch_acc)
        unfamiliar_df.to_csv(os.path.join(folder_name, "Unfamiliar.csv"))