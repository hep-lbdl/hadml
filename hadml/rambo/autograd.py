# Code from https://github.com/madgraph-ml/madspace

import torch
from torch import Tensor
from hadml.rambo.methods import newton
from hadml.rambo.rambo_functions import (
    func_polynomial,
    dfunc_polynomial,
    func_mass,
    dxifunc_mass,
    dpfunc_mass,
)


class RootFinderPolynomial(torch.autograd.Function):
    """
    Custom autograd for our numeric rootfinder which should not loop
    over the iterations in the forward pass.
    """

    @staticmethod
    def forward(ctx, xs: Tensor):
        """Forward pass of rootfinder for rambo polynomial

        Args:
            xs (Tensor): random number input with shape=(b, nparticles - 2)
        """
        # Solve equation numerically for all directly
        nparticles = xs.shape[1] + 2
        func = lambda x: func_polynomial(x, nparticles, xs)
        df = lambda x: dfunc_polynomial(x, nparticles)
        guess = 0.5 * torch.ones_like(xs)
        u = newton(func, df, 0.0, 1.0, guess)

        drdu = df(u)
        ctx.save_for_backward(drdu)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        """
        Make use of inverse function rule [1] to easily get the gradient

        [1] https://en.wikipedia.org/wiki/Inverse_function_rule
        """
        (drdu,) = ctx.saved_tensors
        return grad_output / drdu


class RootFinderMass(torch.autograd.Function):
    """
    Custom autograd for our numeric rootfinder which should not loop
    over the iterations in the forward pass.
    """

    @staticmethod
    def forward(ctx, p0: Tensor, mass: Tensor):
        """Forward pass of rootfinder for massive rambo

        Args:
            p0 (Tensor): energie input with shape=(b, nparticles)
            mass (Tensor): masses with shape=(1, nparticles)
        """
        # solve for xi in massive case, see Ref. [1]
        e_cm = p0.sum(dim=1)
        func = lambda x: func_mass(x, p0, mass, e_cm)
        dxif = lambda x: dxifunc_mass(x, p0, mass)
        dpf = lambda x: dpfunc_mass(x, p0, mass)
        guess = 0.5 * torch.ones((p0.shape[0],))
        xi = newton(func, dxif, 0.0, 1.0, guess)

        dfdxi = dxif(xi)
        dfdp = dpf(xi)
        ctx.save_for_backward(dfdxi, dfdp)
        return xi

    @staticmethod
    def backward(ctx, grad_output):
        dfdxi, dfdp = ctx.saved_tensors
        # Formula: https://en.wikipedia.org/wiki/Implicit_function
        # dx/dp = (df/dx)^(-1) * (df/dp) * (-1)
        dxidp = -dfdp / dfdxi[:, None]
        return grad_output[:, None] * dxidp, None
