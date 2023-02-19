"""Residual Network (ResNet) implementation Based on MLPs."""
try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
            )

    def forward(self, x) -> torch.Tensor:
        return self.model(x) + x
    

def build_layers(input_dim, hidden_dims, output_dim):
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
    for l0, l1 in pairwise(hidden_dims):
        layers.append(ResidualBlock(l0))
        layers.append(nn.Linear(l0, l1))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return layers

  
class ResMLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout = 0.0,         # not used.
                 layer_norm = True,    # not used.
                 last_activation = None):
        super().__init__()
        
        layers = build_layers(input_dim, hidden_dims, output_dim)
        if last_activation is not None:
            layers.append(last_activation)

        self.model = nn.Sequential(*layers)
        
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
