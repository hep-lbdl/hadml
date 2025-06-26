# Code from https://github.com/madgraph-ml/madspace

""" Kinematic functions needed for phase-space mappings """

import torch
from torch import Tensor, cos, sin, cosh, sinh, sqrt, log, atan2
from math import pi


DTYE = torch.get_default_dtype()
EPS = 1e-12 if DTYE == torch.float64 else 1e-6


def kaellen(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Definition of the standard kaellen function [1]

    [1] https://en.wikipedia.org/wiki/Källén_function

    Args:
        x (Tensor): input 1
        y (Tensor): input 2
        z (Tensor): input 3

    Returns:
        Tensor: Kaellen function
    """
    return (x - y - z) ** 2 - 4 * y * z


def rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs rotation around y- and z-axis:

        p -> p' = R_z(phi).R_y(theta).p

    with the explizit matrice following the conventions in [1]
    to achieve proper spherical coordinates [2]:

    R_z = (  1       0         0      0  )
          (  0   cos(phi)  -sin(phi)  0  )
          (  0   sin(phi)   cos(phi)  0  )
          (  0       0         0      1  )

    R_y = (  1       0       0      0       )
          (  0   cos(theta)  0  sin(theta)  )
          (  0       0       1      0       )
          (  0  -sin(theta)  0  cos(theta)  )

    For a 3D vector v = (0, 0, |v|)^T this results in the general spherical
    coordinate vector

        v -> v' = (  |v|*sin(theta)*cos(phi)  )
                  (  |v|*sin(theta)*sin(phi)  )
                  (  |v|*cos(theta)           )

    [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    [2] https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Args:
        p (Tensor): 4-momentum to rotate with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)

    # Define the rotation
    q0 = p[..., 0]
    q1 = (
        p[..., 1] * costheta * cos(phi)
        + p[..., 3] * sintheta * cos(phi)
        - p[..., 2] * sin(phi)
    )
    q2 = (
        p[..., 1] * costheta * sin(phi)
        + p[..., 3] * sintheta * sin(phi)
        + p[..., 2] * cos(phi)
    )
    q3 = p[..., 3] * costheta - p[:, 1] * sintheta

    return torch.stack((q0, q1, q2, q3), dim=-1)


def inv_rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs inverse rotation around y- and z-axis:

        p' -> p = R_y(-theta).R_z(-phi).p

    Args:
        p (Tensor): rotated 4-momentum inverse with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)

    # Define the rotation
    q0 = p[..., 0]
    q1 = (
        p[..., 1] * costheta * cos(phi)
        + p[..., 2] * costheta * sin(phi)
        - p[..., 3] * sintheta
    )
    q2 = p[..., 2] * cos(phi) - p[:, 1] * sin(phi)
    q3 = (
        p[..., 3] * costheta
        + p[..., 1] * sintheta * cos(phi)
        + p[..., 2] * sintheta * sin(phi)
    )

    return torch.stack((q0, q1, q2, q3), dim=-1)


def rapidity(p: Tensor) -> Tensor:
    """Gives the rapidity of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    Es = p[..., 0]
    Pz = p[..., 3]

    y = 0.5 * log((Es + Pz) / (Es - Pz))
    return torch.where(Es < EPS, 99.0, y)


def eta(p: Tensor) -> Tensor:
    """Gives the pseudo-rapidity (eta) of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    Ps = sqrt(esquare(vec3(p)))
    Pz = p[..., 3]

    eta = 0.5 * log((Ps + Pz) / (Ps - Pz))
    return torch.where(Ps < EPS, 99.0, eta)


def phi(p: Tensor) -> Tensor:
    """Gives the azimuthal phi of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    phi = atan2(p[..., 2], p[..., 1])
    return phi


def pT2(p: Tensor) -> Tensor:
    """Gives the squared pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pt2 = p[..., 1] ** 2 + p[..., 2] ** 2
    return pt2


def pT(p: Tensor) -> Tensor:
    """Gives the pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pT2(p))


def vec3(p: Tensor) -> Tensor:
    """Gives the 3-vector of 4-mometum

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    return p[..., 1:]


def delta_rap(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: delta_y with shape=(b,...)
    """
    dy = rapidity(p) - rapidity(q)
    return torch.abs(dy)


def delta_eta(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-eta between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: delta_y with shape=(b,...)
    """
    deta = eta(p) - eta(q)
    return torch.abs(deta)


def delta_phi(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dphi = torch.abs(phi(p) - phi(q))
    dphi = torch.where(dphi > pi, 2 * pi - dphi, dphi)
    return dphi


def deltaR(p: Tensor, q: Tensor) -> Tensor:
    """Gives DeltaR between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dy = delta_eta(p, q)
    dphi = delta_phi(p, q)
    return sqrt(dy**2 + dphi**2)


def pmag2(p: Tensor) -> Tensor:
    """Gives the squared three-momentum |p_vec|^2

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pmag2 = esquare(p[..., 1:])
    return pmag2


def pmag(p: Tensor) -> Tensor:
    """Gives the absolute three-momentum |p_vec|

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pmag2(p))


def sqrt_shat(p: Tensor) -> Tensor:
    """Gives the center-of-mass energy

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    psum = p.sum(dim=1)
    return mass(psum)


def shat(p: Tensor) -> Tensor:
    """Gives the squared center-of-mass energy

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    psum = p.sum(dim=1)
    return lsquare(psum)


def costheta(p: Tensor) -> Tensor:
    """Gives the costheta angle of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return p[..., 3] / pmag(p)


def minv(p1: Tensor, p2: Tensor) -> Tensor:
    """Gives invariant mass of two momenta

    Args:
        p1 (Tensor): momentum 4-vector with shape shape=(b,4)
        p2 (Tensor): momentum 4-vector with shape shape=(b,4)

    Returns:
        Tensor: minv shape=(b,)
    """
    p1p2 = p1 + p2
    return mass(p1p2)


def mass(a: Tensor) -> Tensor:
    """Gives the mass of a particle

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(torch.clip(lsquare(a), min=0.0))


def lsquare(a: Tensor) -> Tensor:
    """Gives the lorentz invariant a^2 using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    a2 = a.square()
    s = a2[..., 0] - a2[..., 1] - a2[..., 2] - a2[..., 3]
    return s


def ldot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the Lorentz inner product ab using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)
        b (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    ab = a * b
    return ab[..., 0] - ab[..., 1] - ab[..., 2] - ab[..., 3]


def esquare(a: Tensor) -> Tensor:
    """Gives the euclidean square a^2 using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Square with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, a)


def edot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the euclidean inner product ab using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)
        b (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, b)


def boost(k: Tensor, p_boost: Tensor, inverse: bool = False) -> Tensor:
    """
    Boost k into the frame of p_boost in argument.
    This means that the following command, for any vector k=(E, px, py, pz)
    gives:

        k  -> k' = boost(k, k, inverse=True) = (M,0,0,0)
        k' -> k  = boost(k', k) = (E, px, py, pz)

    Args:
        k (Tensor): input vector with shape=(b,n,4)/(b,4)
        p_boost (Tensor): boosting vector with shape=(b,1,4)/(b,4)
        inverse (bool): if boost is performed inverse or forward

    Returns:
        k' (Tensor): boosted vector with shape=(b,n,4)/(b,4)
    """
    # Change sign if inverse boost is performed
    sign = -1.0 if inverse else 1.0

    # Perform the boost
    # This is in fact a numerical more stable implementation then often used
    rsq = torch.clip(mass(p_boost), min=EPS)
    k0 = (k[..., 0] * p_boost[..., 0] + sign * edot(k[..., 1:], p_boost[..., 1:])) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + sign * c1 * p_boost[..., 1]
    k2 = k[..., 2] + sign * c1 * p_boost[..., 2]
    k3 = k[..., 3] + sign * c1 * p_boost[..., 3]

    return torch.stack((k0, k1, k2, k3), dim=-1)


def boost_beam(
    q: Tensor,
    rapidity: Tensor,
    inverse: bool = False,
) -> Tensor:
    """Boosts q along the beam axis with given rapidity

    Args:
        q (Tensor): input vector with shape=(b,n,4)/(b,4)
        rapidity (Tensor): boosting parameter with shape=(b,1)/(b,)
        inverse (bool, optional): inverse boost. Defaults to False.

    Returns:
        q' (Tensor): boosted vector with shape=(b,n,4)
    """
    sign = -1.0 if inverse else 1.0

    pi0 = q[..., 0] * cosh(rapidity) + sign * q[..., 3] * sinh(rapidity)
    pix = q[..., 1]
    piy = q[..., 2]
    piz = q[..., 3] * cosh(rapidity) + sign * q[..., 0] * sinh(rapidity)

    return torch.stack((pi0, pix, piy, piz), dim=-1)
