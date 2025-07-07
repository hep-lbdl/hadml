# Code from https://github.com/madgraph-ml/madspace

import torch
from torch import Tensor, sqrt


# -------------- Get polynomial equation and its derivative -----------
#
def func_polynomial(u: Tensor, nparticles: int, xs: Tensor) -> Tensor:
    """
    See algorithm 1 in:
        [1] https://arxiv.org/pdf/1308.2922.pdf
        [2] https://archiv.ub.uni-heidelberg.de/volltextserver/29154/1/thesis_RW.pdf

    Note: There is a factor 2 missing in the exponent in [1]! Has been reported to the author.

    Args:
        u (Tensor): required output with shape=(b, nparticles-2)
        nparticles (int): number of external particles
        xs (Tensor): random numbers with shape=(b, nparticles-2)

    Returns:
        Tensor: function of interest
    """
    i = torch.arange(2, nparticles)[None, :].to(u.device)
    f = (
        (nparticles + 1 - i) * u ** (2 * (nparticles - i))
        - (nparticles - i) * u ** (2 * (nparticles + 1 - i))
        - xs
    )
    return f


def dfunc_polynomial(u: Tensor, nparticles: int) -> Tensor:
    """Gradient of func_polynomial with respect to u"""
    i = torch.arange(2, nparticles)[None, :].to(u.device)
    df = (nparticles + 1 - i) * (2 * (nparticles - i)) * u ** (
        2 * (nparticles - i) - 1
    ) - (nparticles - i) * (2 * (nparticles + 1 - i)) * u ** (
        2 * (nparticles + 1 - i) - 1
    )
    return df


# -------------- Get mass equation and its derivatives -----------
#
def func_mass(xi: Tensor, p0: Tensor, m: Tensor, e_cm: Tensor) -> Tensor:
    """
    See momentum reshuffling in
        [1] Rambo [Comput. Phys. Commun. 40 (1986) 359-373]
        [2] https://arxiv.org/pdf/2305.07696.pdf

    Args:
        xi (Tensor): scaling factor shape=(b,)
        p0 (Tensor): energies with shape=(b, nparticles)
        m (Tensor): particle masses with shape=(1, nparticles)
        e_cm (Tensor): COM energy with shape=(b,)

    Returns:
        Tensor: function of interest
    """
    root = sqrt(xi[:, None] ** 2 * p0**2 + m**2)
    f = torch.sum(root, dim=-1) - e_cm
    return f


def dxifunc_mass(xi: Tensor, p0: Tensor, m: Tensor) -> Tensor:
    """Gradient of func_mass with respect to xi"""
    root = sqrt(xi[:, None] ** 2 * (p0**2).to(xi.device) + (m**2).to(xi.device))
    df = torch.sum(xi[:, None] * (p0**2).to(xi.device) / root, dim=-1).to(xi.device)
    return df


def dpfunc_mass(xi: Tensor, p0: Tensor, m: Tensor) -> Tensor:
    """Gradient of func_mass with respect to all pi"""
    root = sqrt(xi[:, None] ** 2 * (p0**2).to(xi.device) + (m**2).to(xi.device))
    df = xi[:, None] ** 2 * p0.to(xi.device) / root - 1.0
    return df
