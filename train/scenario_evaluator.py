import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from utils.plot import save_plot
from utils.get_loaders import get_loader
from utils.avg_meter import AverageMeter
from collections import defaultdict

class DomainEvaluator:
    def __init__(self, algorithm, device, scenario, configs):
        self.algo = algorithm
        self.device = device
        self.scenario = scenario
        self.configs = configs
        self.acc_matrix = pd.DataFrame(index=scenario, columns=scenario)
        self.loss_avg_meters = defaultdict(lambda: AverageMeter())

    def test_domain(self, test_loader):
        self.algo.feature_extractor.eval()
        self.algo.classifier.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0], data[1]
                x, y = x.to(self.device), y.to(self.device)
                logits = self.algo.classifier(self.algo.feature_extractor(x))
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        accuracy = correct / total
        return accuracy

    def test_all_domain(self):
        acc_dict = defaultdict(float)
        for domain in self.scenario:
            test_loader = get_loader(self.configs.Dataset_Name, domain, "test")
            acc = self.test_domain(test_loader)
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
            bwt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, row] for row in row_names) / T
            bwt_values.append(bwt_t)
        bwt_column = pd.DataFrame(bwt_values, index=self.scenario, columns=['BWT'])
        return bwt_column

    def calc_adapt(self):
        adapt_values = [0.0]  # Adapt does not exist for source model so we start with 0
        source = self.scenario[0]
        for T, domain in enumerate(self.scenario[1:], start=1):  # ! Keep in mind T starts from 1
            row_names = [self.scenario[i] for i in range(1, T + 1)]
            adapt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, source] for row in row_names) / T
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

    def calc_overall_metrics(self):
        self.loss_avg_meters["avg_acc"].update(self.acc.iloc[1:]['ACC'].mean())
        self.loss_avg_meters["avg_bwt"].update(self.bwt.iloc[2:]['BWT'].mean())
        self.loss_avg_meters["avg_adapt"].update(self.adapt.iloc[1:]["Adapt"].mean())
        self.loss_avg_meters["avg_generalise"].update(self.generalise.iloc[1:-1]["Generalise"].mean())

    def get_src_risk(self, domain):
        src_loader = get_loader(self.configs.Dataset_Name, domain, "test")

        self.algo.feature_extractor.eval()
        self.algo.classifier.eval()

        self.loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels, _ in src_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits = self.algo.classifier(self.algo.feature_extractor(inputs))
                batch_loss = self.loss_fn(logits, labels)

                total_loss += batch_loss.item() * labels.size(0)
                total_samples += labels.size(0)

        self.loss_avg_meters["src_risk"].update(total_loss / total_samples)
        # return total_loss / total_samples

    def save(self, folder_name):
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

    def save_plot(self, savefile):
        save_plot(self.acc_matrix, self.scenario, savefile)



# Usage:
# evaluator = DomainEvaluator(algorithm_instance, device, scenario, configs)
# acc_dict = evaluator.test_all_domain()
# evaluator.update('modelName', acc_dict)
# evaluator.calc_metric()
# evaluator.save('folderName')
