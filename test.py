import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import AttentionModel
from tsp import generate_instances, evaluate


model = AttentionModel(2, 5, 5, 3, 3, 10)

instances = generate_instances(5, 2)

print('-' * 15 + 'sampling' + '-' * 15)
for i in range(3):
    solutions, log_p = model(instances, 'sampling')
    print(solutions)
    print(evaluate(instances, solutions).sum().item())

print('-' * 16 + 'greedy' + '-' * 16)
solutions, log_p = model(instances)
print(solutions)
print(evaluate(instances, solutions).sum().item())
