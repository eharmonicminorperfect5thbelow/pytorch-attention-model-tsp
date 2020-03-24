import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_instances(size, num):
    return torch.rand(num, size, 2)

def evaluate(instances, solutions):
    num = instances.size()[0]
    size = instances.size()[1]
    score = torch.zeros(num)

    for i in range(num):
        for j in range(size - 1):
            a = instances[i][solutions[i][j]]
            b = instances[i][solutions[i][j + 1]]
            score[i] += torch.sqrt(((a - b) * (a - b)).sum())

        a = instances[i][solutions[i][0]]
        b = instances[i][solutions[i][size - 1]]
        score[i] += torch.sqrt(((a - b) * (a - b)).sum())

    return score

def plot(instance, solution):
    ins = instance.numpy().T
    sol = solution.numpy()
    sol = np.concatenate([sol, [sol[0]]], 0)
    coo = ins[:, sol]

    plt.plot(coo[0], coo[1])
    plt.show()
