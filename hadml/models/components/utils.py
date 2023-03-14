import torch
import torch.nn as nn
from torch import Tensor
import math

def inv_boost(cond_info: Tensor, output_info: Tensor):

    device = cond_info.device

    cond_info_max = torch.tensor([50., 50., 50., 50]).to(device)
    cond_info_min = torch.tensor([0., -50., -50., -50]).to(device)
    cond_info = (cond_info + 1)/2.*(cond_info_max-cond_info_min)+cond_info_min
    E0, p0 = cond_info[:,0:1], cond_info[:,1:]
    sum_p2 = torch.sum(p0**2, dim=1).reshape(-1, 1)
    P0 = torch.sqrt(sum_p2)
    hello = E0**2 - sum_p2
    m0 = torch.sqrt(E0**2 - sum_p2)
    gamma = E0 / m0

    velocity = p0 / E0
    v0_mag = P0 / E0
    n0 = p0 / P0

    out_truth_max = torch.tensor([math.pi/2, math.pi]).to(device)
    out_truth_min = torch.tensor([-math.pi/2, 0]).to(device)
    output_info = (output_info + 1)/2.*(out_truth_max-out_truth_min)+out_truth_min
    phi = output_info[:,0].reshape(-1, 1)
    theta = output_info[:,1].reshape(-1, 1)
    E_prime = m0 / 2
    P_prime = torch.sqrt(E_prime**2 - 0.0013957**2)
    p1_prime = torch.cat((torch.sin(theta)*torch.sin(phi), torch.sin(theta)*torch.cos(phi), torch.cos(theta)), dim=1) * P_prime
    p2_prime = - p1_prime

    def inv_boost_fn(p_prime: Tensor):
        """4vecot [E, px, py, pz] in boost frame (aka cluster frame)"""
        n_dot_p = torch.sum((n0 * p_prime), dim=1).reshape(-1, 1)
        E = gamma * (E_prime + v0_mag * n_dot_p)
        p = p_prime + (gamma - 1) * n_dot_p * n0 + p0 / 2
        return E, p

    E1, p1 = inv_boost_fn(p1_prime)
    E2, p2 = inv_boost_fn(p2_prime)

    out_info = torch.cat((E1, p1, E2, p2), dim=1)

    out_info_max = torch.tensor([40., 30., 30., 30, 40., 30., 30., 30]).to(device)
    out_info_min = torch.tensor([0., -30., -30., -30., 0., -30., -30., -30.]).to(device)
    out_info = (out_info - out_info_min)/(out_info_max - out_info_min)*2 - 1

    return out_info