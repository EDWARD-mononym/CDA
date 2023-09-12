from collections import defaultdict
import pandas as pd
import torch

from utils.get_loaders import get_loader

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
        acc_column = self.calc_ACC()
        bwt_column = self.calc_BWT()
        adapt_column = self.calc_adapt()
        generalise_column = self.calc_generalise()
        self.acc_matrix = pd.concat([self.acc_matrix, acc_column, bwt_column, adapt_column, generalise_column], axis=1)

    def calc_ACC(self):
        acc_values = []
        for T, domain in enumerate(self.scenario): #! Keep in mind T starts from 0
            acc_t = self.acc_matrix[domain].iloc[0:(T + 1)].sum() / (T + 1)
            acc_values.append(acc_t)
        acc_column = pd.DataFrame(acc_values, index=self.scenario, columns=['ACC'])
        return acc_column

    def calc_BWT(self):
        bwt_values = [0.0] # BWT does not exist for 0, so we start with 0
        for T, domain in enumerate(self.scenario[1:], start=1): #! Keep in mind T starts from 1
            row_names = [self.scenario[i] for i in range(0, T + 1)]
            bwt_t = sum(self.acc_matrix.loc[row, domain] - self.acc_matrix.loc[row, row] for row in row_names) / T
            bwt_values.append(bwt_t)
        bwt_column = pd.DataFrame(bwt_values, index=self.scenario, columns=['BWT'])
        return bwt_column
    
    def calc_adapt(self):
        adapt_values = [0.0] # Adapt does not exist for 0, so we start with 0
        for T, domain in enumerate(self.scenario[1:], start=1): #! Keep in mind T starts from 1
            row_names = [self.scenario[i] for i in range(1, T+1)]
            prev_row_names = [self.scenario[i] for i in range(0, T)]
            adapt_t = sum(self.acc_matrix.loc[row, row] - self.acc_matrix.loc[row, prev_column] for row, prev_column in zip(row_names, prev_row_names)) / T
            adapt_values.append(adapt_t)
        adapt_column = pd.DataFrame(adapt_values, index=self.scenario, columns=['Adapt'])
        return adapt_column

    def calc_generalise(self):
        generalise_values = [0.0] # Generalise does not exist for 0, so we start with 0
        for T, domain in enumerate(self.scenario[1:-1], start=1): #! Keep in mind T starts from 1 and ends before last domain
            next_row_names = [self.scenario[i] for i in range(2, T + 2)]
            current_column_names = [self.scenario[i] for i in range(1, T + 1)]
            generalise_t = sum(self.acc_matrix.loc[next_row, column] - self.acc_matrix.loc[next_row, self.scenario[0]] for next_row, column in zip(next_row_names, current_column_names)) / T
            generalise_values.append(generalise_t)
        generalise_values.append(0.0) # Add another 0 for end of domain where Generalise does not exist
        generalise_column = pd.DataFrame(generalise_values, index=self.scenario, columns=['Generalise'])
        return generalise_column