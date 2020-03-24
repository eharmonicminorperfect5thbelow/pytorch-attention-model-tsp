import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from layers import AttentionModel
from tsp import generate_instances


model = AttentionModel(2, 5, 5, 3, 3)

instances = generate_instances(5, 2)

solutions, log_p = model(instances, 'sampling')

print(solutions)
