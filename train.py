import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import AttentionModel
from tsp import generate_instances, evaluate, plot


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

# model = AttentionModel(2, 5, 5, 3, 3)
base_model = try_gpu(AttentionModel(2,16,32,3,3))
model = try_gpu(AttentionModel(2,16,32,3,3,100))
optimizer = optim.Adam(model.parameters(), lr=0.001)

problem = try_gpu(generate_instances(10, 2))
selected, log_p = model(problem)
cost = evaluate(problem, selected)
print(cost.sum())

for e in range(100):
    print('Epoch -', e)

    cost_total = 0
    base_cost_total = 0
    
    for i in tqdm(range(10)):
        x = try_gpu(generate_instances(10, 100))

        selected, log_p = model(x, 'sampling')
        cost = try_gpu(evaluate(x, selected))

        base_selected, base_log_p = base_model(x)
        base_cost = try_gpu(evaluate(x, base_selected))

        cost_total += cost.sum()
        base_cost_total += base_cost.sum()

        loss = ((cost - base_cost) * log_p).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if cost_total < base_cost_total:
        base_model.load_state_dict(model.state_dict())

    current_selected, current_log_p = model(problem)
    current_cost = evaluate(problem, current_selected)
    print(current_cost.sum())

# plot(x[0], selected[0])