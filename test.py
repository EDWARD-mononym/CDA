from itertools import product

hyperparameters = {
    'lr': [0.01, 0.001],
    'hidden_dim': [128, 256],
    'gamma': [0.5, 0.8, 0.6]
}

hyperparameter_combinations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

print(hyperparameter_combinations[0])