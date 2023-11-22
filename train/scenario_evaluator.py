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
        self.avg_ACC = {col: AverageMeter() for col in self.acc_matrix.columns}
        self.avg_BWT = {col: AverageMeter() for col in self.acc_matrix.columns}
        self.avg_Adapt = {col: AverageMeter() for col in self.acc_matrix.columns}
        self.avg_Generalise = {col: AverageMeter() for col in self.acc_matrix.columns}

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

    def calc_metric(self):  # Calculate ACC, BWT, Adapt, Generalise
        self.acc = self.calc_ACC()
        self.bwt = self.calc_BWT()
        self.adapt = self.calc_adapt()
        self.generalise = self.calc_generalise()

        # self.acc_matrix = pd.concat([self.acc_matrix, acc_column, bwt_column, adapt_column, generalise_column], axis=1)

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

    def calc_adapt(self):
        adapt_values = [0.0]  # Adapt does not exist for source model so we start with 0
        for T, domain in enumerate(self.scenario[1:], start=1):  # ! Keep in mind T starts from 1
            row_names = [self.scenario[i] for i in range(1, T + 1)]
            # adapt_t = sum(self.acc_matrix.loc[row, row] - self.acc_matrix.loc[row, row_names[i-1]] for i, row in enumerate(row_names)) / T
            adapt_t = sum(self.acc_matrix.loc[row, row] for i, row in enumerate(row_names)) / T
            adapt_values.append(adapt_t)
        adapt_column = pd.DataFrame(adapt_values, index=self.scenario, columns=['Adapt'])
        return adapt_column

    def calc_generalise(self):
        generalise_values = [0.0]  # Generalise does not exist for source model so we start with 0
        for T, domain in enumerate(self.scenario[1:-1],
                                   start=1):  # ! Keep in mind T starts from 1 and ends before last domain
            unseen_rows = [self.scenario[i] for i in range(T + 1, len(self.scenario))]
            n_unseen = len(unseen_rows)
            generalise_t = sum(
                self.acc_matrix.loc[unseen, domain] - self.acc_matrix.loc[unseen, self.scenario[0]] for unseen in
                unseen_rows) / n_unseen
            generalise_values.append(generalise_t)
        generalise_values.append(0.0)  # Add another 0 for end of domain where Generalise does not exist
        generalise_column = pd.DataFrame(generalise_values, index=self.scenario, columns=['Generalise'])
        return generalise_column

    def calc_overall_metrics(self, loss_avg_meters):
        loss_avg_meters["avg_acc"].update(self.acc.iloc[1:]['ACC'].mean())
        loss_avg_meters["avg_bwt"].update(self.bwt.iloc[2:]['BWT'].mean())
        loss_avg_meters["avg_adapt"].update(self.adapt.iloc[1:]["Adapt"].mean())
        loss_avg_meters["avg_generalise"].update(self.generalise.iloc[1:-1]["Generalise"].mean())

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
        with open(f"{folder_name}_Overall.csv", 'w') as f:
            # Summarise the metrics
            summary = {
                "avg_acc": [self.acc.iloc[1:]['ACC'].mean()],
                "avg_bwt": [self.bwt.iloc[2:]['BWT'].mean()],
                "avg_adapt": [self.adapt.iloc[1:]["Adapt"].mean()],
                "avg_generalise": [self.generalise.iloc[1:-1]["Generalise"].mean()]}
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(f, index=False)

        with open(f"{folder_name}_ModelPerformance.csv", 'w') as f:
            for model in self.acc_matrix.columns:  # Save performance of each model
                # Separate performance of each model to make it easier to process
                column_df = self.acc_matrix[model]
                row_df = pd.DataFrame(column_df).T
                # Append metrics into the row df
                row_df["ACC"] = self.acc.loc[model, "ACC"]
                row_df["BWT"] = self.bwt.loc[model, "BWT"]
                row_df["Adapt"] = self.adapt.loc[model, "Adapt"]
                row_df["Generalise"] = self.generalise.loc[model, "Generalise"]

                # Save model performance
                f.write(f"Model {model} performance")
                row_df.to_csv(f, index=True)
                f.write("\n")

        with open(f"{folder_name}_Rmatrix.csv", 'w') as f:
            f.write("R matrix")
            # Save R matrix
            self.acc_matrix.to_csv(f, index=True)

    def update_overall(self):
        for col_id, col in enumerate(self.acc_matrix.columns):
            for row in self.acc_matrix.index:
                self.avg_Rmatrix[(row, col)].update(self.acc_matrix.loc[row, col])

            self.avg_ACC[col].update(self.acc.loc[col]['ACC'])
            self.avg_BWT[col].update(self.bwt.loc[col]['BWT'])
            self.avg_Adapt[col].update(self.adapt.loc[col]['Adapt'])
            self.avg_Generalise[col].update(self.generalise.loc[col]['Generalise'])

    def save_overall(self, folder_name):
        overall_df = pd.DataFrame(columns=self.acc_matrix.columns, index=self.acc_matrix.index)
        metrics_df = pd.DataFrame(columns = ['ACC', 'BWT', 'Adapt', 'Generalise'], index=self.acc_matrix.index)

        for (row, col) in self.avg_Rmatrix:
            overall_df.at[row, col] = f"{self.avg_Rmatrix[(row, col)].average()* 100:.2f} ± {self.avg_Rmatrix[(row, col)].standard_deviation()* 100:.2f}"

        for row in self.acc_matrix.index:
            metrics_df.at[row, 'ACC'] = f"{self.avg_ACC[row].average()* 100:.2f} ± {self.avg_ACC[row].standard_deviation()* 100:.2f}"
            metrics_df.at[row, 'BWT'] = f"{self.avg_BWT[row].average()* 100:.2f} ± {self.avg_BWT[row].standard_deviation()* 100:.2f}"
            metrics_df.at[row, 'Adapt'] = f"{self.avg_Adapt[row].average()* 100:.2f} ± {self.avg_Adapt[row].standard_deviation()* 100:.2f}"
            metrics_df.at[row, 'Generalise'] = f"{self.avg_Generalise[row].average()* 100:.2f} ± {self.avg_Generalise[row].standard_deviation()* 100:.2f}"

        overall_df.to_csv(f"{folder_name}_R_matrix.csv")
        metrics_df.to_csv(f"{folder_name}_Metrics.csv")

    def save_plot(self, savefile):
        save_plot(self.acc_matrix, self.scenario, savefile)



# Usage:
# evaluator = DomainEvaluator(algorithm_instance, device, scenario, configs)
# acc_dict = evaluator.test_all_domain()
# evaluator.update('modelName', acc_dict)
# evaluator.calc_metric()
# evaluator.save('folderName')
