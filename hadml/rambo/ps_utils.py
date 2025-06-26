# Code from https://github.com/madgraph-ml/madspace

""" Helper functions needed for phase-space mappings """

import torch
from torch import Tensor, sqrt, log, cos, sin, sqrt
from hadml.rambo.kinematics import kaellen, pi


def two_particle_density(s: Tensor, p1_2: Tensor, p2_2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.8) in [1]

    Args:
        s (Tensor): squared COM energy of the proces with shape=(b,)
        p1_2 (Tensor): Mass/virtualiy of outgoing particle 1 with shape=(b,)
        p2_2 (Tensor): Mass/virtualiy of outgoing particle 1 with shape=(b,)

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    # No overall (2*pi)^(-2) here!
    g2 = sqrt(kaellen(s, p1_2, p2_2)) / (8 * s)
    return g2


def three_particle_density() -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (G.12) in [2]

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    # No overall (2*pi)^(-5) here!
    g2 = torch.tensor(1 / 8.0)
    return g2


def tinv_two_particle_density(s: Tensor, p1_2: Tensor, p2_2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.22) in [1]

    Args:
        s (Tensor): squared COM energy of the proces with shape=(b,)
        p1_2 (Tensor): Virtuality of incoming particle 1 with shape=(b,)
        p2_2 (Tensor): Virtuality of incoming particle 2 with shape=(b,)

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    # No overall (2*pi)^(-2) here!
    g = 1 / (4 * sqrt(kaellen(s, p1_2, p2_2)))
    return g


def build_p_in(e_cm: Tensor) -> Tensor:
    """Build symmetric incoming momenta given the center of mass energy

    Args:
        e_cm: Center of mass energy, shape=(b,)

    Returns:
        p_in: Incoming momenta, shape=(b,2,4)
    """
    zeros = torch.zeros_like(e_cm)
    p_cms = e_cm / 2
    p1 = torch.stack([p_cms, zeros, zeros, p_cms], dim=1)
    p2 = torch.stack([p_cms, zeros, zeros, -p_cms], dim=1)
    p_in = torch.stack([p1, p2], dim=1)
    return p_in


def pin_to_x1x2(p_in: Tensor, e_cm: Tensor) -> Tensor:
    """Calculates the pdf fractions x1, x2 from the initial state
    momenta in their lab frame

    Args:
        p_in (Tensor): initial state momenta in lab frame with shape=(b,2,4)
        e_cm (Tensor): center of mass energy with shape=(b,)

    Returns:
        x1x2 (Tensor): pdf fractions with shape=(b,2)
    """
    pp = p_in[:, 0, 0] * 2
    pm = p_in[:, 1, 0] * 2

    # Get the bjorken variables
    x1 = pp / e_cm
    x2 = pm / e_cm

    x1x2 = torch.stack([x1, x2], dim=1)
    return x1x2


def map_fourvector_rambo(r: Tensor) -> Tensor:
    """Transform unit hypercube points into into four-vectors.

    Args:
        r (Tensor): 4n random numbers with shape=(b,n,4)

    Returns:
        q (Tensor): n 4-Momenta with shape=(b,n,4)
    """
    costheta = 2.0 * r[:, :, 0] - 1.0
    phi = 2.0 * pi * r[:, :, 1]

    q0 = -log(r[:, :, 2] * r[:, :, 3])
    qx = q0 * sqrt(1 - costheta**2) * cos(phi)
    qy = q0 * sqrt(1 - costheta**2) * sin(phi)
    qz = q0 * costheta

    return torch.stack([q0, qx, qy, qz], dim=-1)


def map_fourvector_rambo_diet(q0: Tensor, costheta: Tensor, phi: Tensor) -> Tensor:
    """Transform energies and angles into proper 4-momentum
    Needed for rambo on diet

    Args:
        q0 (Tensor): energy with shape=(b,n-1)
        costheta (Tensor): costheta angle with shape shape=(b,n-1)
        phi (Tensor): azimuthal angle with sshape=(b,n-1)

    Returns:
        q (Tensor): n 4-Momenta with shape=(b,n-1,4)
    """
    qx = q0 * sqrt(1 - costheta**2) * cos(phi)
    qy = q0 * sqrt(1 - costheta**2) * sin(phi)
    qz = q0 * costheta

    return torch.stack([q0, qx, qy, qz], dim=-1)


def two_body_decay_factor(
    M_i_minus_1: Tensor,
    M_i: Tensor,
    m_i_minus_1: Tensor,
) -> Tensor:
    """Gives two-body decay factor from recursive n-body phase space"""
    return (
        1.0
        / (8 * M_i_minus_1**2)
        * sqrt(
            (M_i_minus_1**2 - (M_i + m_i_minus_1) ** 2)
            * (M_i_minus_1**2 - (M_i - m_i_minus_1) ** 2)
        )
    )
