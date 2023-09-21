from collections import defaultdict
import pandas as pd
import torch

from utils.get_loaders import get_loader
from utils.plot import save_plot

def test_domain(test_loader, feature_extractor, classifier, device):
    feature_extractor.eval()
    classifier.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)
            logits = classifier(feature_extractor(x))
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    accuracy = correct / total
    return accuracy

def test_all_domain(datasetname, scenario, feature_extractor, classifier, device):
    acc_dict = defaultdict(float)
    for domain in scenario:
        testloader = get_loader(datasetname, domain, "test")
        acc = test_domain(testloader, feature_extractor, classifier, device)
        acc_dict[domain] = acc
    return acc_dict

class Acc_matrix():
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.acc_matrix = pd.DataFrame(index=scenario, columns=scenario)

    def update(self, model, acc_dict):
        self.acc_matrix[model] = self.acc_matrix.index.map(acc_dict)

    def calc_metric(self): # Calculate ACC, BWT, Adapt, Generalise
        self.acc = self.calc_ACC()
        self.bwt = self.calc_BWT()
        self.adapt = self.calc_adapt()
        self.generalise = self.calc_generalise()
        # self.acc_matrix = pd.concat([self.acc_matrix, acc_column, bwt_column, adapt_column, generalise_column], axis=1)

    def calc_ACC(self):
        acc_values = [0.0] # ACC not counted for souce model, so we start with 0
        for T, domain in enumerate(self.scenario[1:], start=1): #! Keep in mind T starts from 1
            acc_t = (1/T) * self.acc_matrix[domain].iloc[1:(T + 1)].sum()
            acc_values.append(acc_t)
        acc_column = pd.DataFrame(acc_values, index=self.scenario, columns=['ACC'])
        return acc_column

    def calc_BWT(self):
        bwt_values = [0.0, 0.0] # BWT does not exist for source model and first target, so we start with two 0
        for T, domain in enumerate(self.scenario[2:], start=2): #! Keep in mind T starts from 2
            row_names = [self.scenario[i] for i in range(1, T + 1)]
            bwt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, row] for row in row_names) / T
            bwt_values.append(bwt_t)
        bwt_column = pd.DataFrame(bwt_values, index=self.scenario, columns=['BWT'])
        return bwt_column
    
    def calc_adapt(self):
        adapt_values = [0.0] # Adapt does not exist for source model so we start with 0
        source = self.scenario[0]
        for T, domain in enumerate(self.scenario[1:], start=1): #! Keep in mind T starts from 1
            row_names = [self.scenario[i] for i in range(1, T + 1)]
            adapt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, source] for row in row_names) / T
            adapt_values.append(adapt_t)
        adapt_column = pd.DataFrame(adapt_values, index=self.scenario, columns=['Adapt'])
        return adapt_column

    def calc_generalise(self):
        generalise_values = [0.0] # Generalise does not exist for source model so we start with 0
        for T, domain in enumerate(self.scenario[1:-1], start=1): #! Keep in mind T starts from 1 and ends before last domain
            unseen_rows = [self.scenario[i] for i in range(T+1, len(self.scenario))]
            n_unseen = len(unseen_rows)
            generalise_t = sum(self.acc_matrix.loc[unseen, domain] - self.acc_matrix.loc[unseen, self.scenario[0]] for unseen in unseen_rows) / n_unseen
            generalise_values.append(generalise_t)
        generalise_values.append(0.0) # Add another 0 for end of domain where Generalise does not exist
        generalise_column = pd.DataFrame(generalise_values, index=self.scenario, columns=['Generalise'])
        return generalise_column

    def save(self, file_name):
        with open(file_name, 'w') as f:
            # Summarise the metrics
            summary = {
            "avg_acc": [self.acc.iloc[1:]['ACC'].mean()],
            "avg_bwt" : [self.bwt.iloc[2:]['BWT'].mean()],
            "avg_adapt" : [self.adapt.iloc[1:]["Adapt"].mean()],
            "avg_generalise" : [self.generalise.iloc[1:-1]["Generalise"].mean()]}
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(f, index=False)

            for model in self.acc_matrix.columns: # Save performance of each model 
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

            f.write("R matrix")
            # # Save R matrix
            self.acc_matrix.to_csv(f, index=True)

    def save_plot(self, savefile):
        save_plot(self.acc_matrix, self.scenario, savefile)