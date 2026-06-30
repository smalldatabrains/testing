import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

# Model Visualization
from torchinfo import summary

# Use of dataset and Dataloader for better GPU optimization
from torch.utils.data import TensorDataset, DataLoader

import math

# Denoisig Diffusion Probabilistics Models implementation [2020]

# original data
x0 = []
for t in range (0, 4):
    x0.append(math.sin(t))

# x1 is x0 + a little bit of noise (x2 is even more noise etc ... till xn which is pure Gaussian noise). x0:T means x0 to xT, x1:T means x1 to xT

# example
# x₀ = cat image
# x₁ = slightly noisy cat
# x₂ = noisy cat
# x₃ = very noisy cat
# x₄ = almost random noise

