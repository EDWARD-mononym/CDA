import random
import numpy as np
import torch

def accuracy(pred, y):
    correct = (pred == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)