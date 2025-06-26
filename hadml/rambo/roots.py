# Code from https://github.com/madgraph-ml/madspace

from torch import Tensor
from hadml.rambo.autograd import RootFinderPolynomial, RootFinderMass


def get_u_parameter(xs: Tensor) -> Tensor:
    """Returns the solution of the equation
    ...math::
        r_{i-1} =(n+1-i)*u_i^{2n-2i} - (n-i)*u_i^{2n+2-2i} for u_i

    for all ``i`` in {2,nparticles}

    Args:
        xs (Tensor): Random number input with shape=(b, nparticles - 2)

    Returns:
        u_i (Tensor): solution with shape=(b, nparticles - 2)
    """
    return RootFinderPolynomial.apply(xs)


def get_xi_parameter(p0: Tensor, mass: Tensor) -> Tensor:
    """Returns the solution of the equation
    ...math::
        e_cm = sum_i sqrt(m_i^2 + xi^2 * p0_i^2) for xi

    Args:
        p0 (Tensor): energies with shape=(b, nparticles)
        m (Tensor): particle masses with shape=(1, nparticles)

    Returns:
        xi (Tensor): scaling parameter with shape=(b,)
    """
    return RootFinderMass.apply(p0, mass)
