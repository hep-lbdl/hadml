from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def InvsBoost(
    cluster: Tensor,
    angles: Tensor,
    mass_1: float = 0.0013957,
    mass_2: float = 0.0013957,
) -> Tensor:
    """Boost the kinematics back to the lab frame

    Args:
        cluster: 4 momentum of the cluster in the lab frame [E, px, py, pz]
        angles: angles of the outgoing particles in the CM frame [phi, theta]
        mass_1: rest mass of the first outgoing particle in GeV
        mass_2: rest mass of the second outgoing particle in GeV

    Returns:
        4 momentum of the two outgoing particles in the lab frame
    """

    device = cluster.device

    E0, p0 = cluster[:, 0:1], cluster[:, 1:]
    sum_p2 = torch.sum(p0**2, dim=1).reshape(-1, 1)
    P0 = torch.sqrt(sum_p2)
    m0 = torch.sqrt(E0**2 - sum_p2)
    gamma = E0 / m0

    v0_mag = P0 / E0
    n0 = p0 / P0

    phi = angles[:, 0].reshape(-1, 1)
    theta = angles[:, 1].reshape(-1, 1)
    E_prime = (m0**2 + mass_1**2 - mass_2**2) / 2 / m0
    P_prime = torch.sqrt(E_prime**2 - mass_1**2)
    p1_prime = (
        torch.cat(
            (
                torch.sin(theta) * torch.sin(phi),
                torch.sin(theta) * torch.cos(phi),
                torch.cos(theta),
            ),
            dim=1,
        )
        * P_prime
    )
    p2_prime = -p1_prime

    def inv_boost_fn(p_prime: Tensor):
        """4vecot [E, px, py, pz] in boost frame (aka cluster frame)"""
        n_dot_p = torch.sum((n0 * p_prime), dim=1).reshape(-1, 1)
        E = gamma * (E_prime + v0_mag * n_dot_p)
        p = p_prime + (gamma - 1) * n_dot_p * n0 + p0 / 2
        return E, p

    E1, p1 = inv_boost_fn(p1_prime)
    E2, p2 = inv_boost_fn(p2_prime)

    hadrons = torch.cat((E1, p1, E2, p2), dim=1)

    return hadrons


class NormModule(nn.Module):
    def __init__(self, val_max: List[float], val_min: List[float]):
        super().__init__()
        self.register_buffer("val_max", torch.tensor(val_max))
        self.register_buffer("val_min", torch.tensor(val_min))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.val_min) / (self.val_max - self.val_min) * 2.0 - 1.0


class InvsNormModule(nn.Module):
    def __init__(self, val_max: List[float], val_min: List[float]):
        super().__init__()
        self.register_buffer("val_max", torch.tensor(val_max))
        self.register_buffer("val_min", torch.tensor(val_min))

    def forward(self, x: Tensor) -> Tensor:
        return (x + 1.0) / 2.0 * (self.val_max - self.val_min) + self.val_min
