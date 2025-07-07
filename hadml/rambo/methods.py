# Code from https://github.com/madgraph-ml/madspace

""" Numerical methods to find root"""

from typing import Callable, Optional
import torch
from torch import Tensor

DEFAULT_FLOAT = torch.get_default_dtype()
ITER = 100
XTOL = 2 * 1e3 * torch.finfo(DEFAULT_FLOAT).eps
RTOL = 4 * torch.finfo(DEFAULT_FLOAT).eps


def newton(
    f: Callable,
    df: Callable,
    a: float,
    b: float,
    x0: Tensor = None,
    max_iter: int = ITER,
    epsilon=1e-12,
):
    # Define lower/upper boundaries as tensor
    xa = a * torch.ones_like(x0)
    xb = b * torch.ones_like(x0)

    if torch.any(f(xa) * f(xb) > 0):
        raise ValueError(f"None or no unique root in given intervall [{a},{b}]")

    for _ in range(max_iter):
        # do newtons-step but make sure gradient is not too small
        df0 = torch.where(df(x0) < epsilon, epsilon, df(x0)).to(x0.device)
        x1 = x0 - f(x0).to(x0.device) / df0

        # check if within given intervall
        higher = x1 > xb
        lower = x1 < xa
        if torch.any(higher):
            x1[higher] = (xb[higher] + x0[higher]) / 2
        if torch.any(lower):
            x1[lower] = (xa[lower] + x0[lower]) / 2

        if torch.allclose(x0, x1, atol=XTOL, rtol=RTOL):
            # print(f"coverged after {i+1} iterations")
            return x1

        # Adjust brackets
        low = f(x1).to(x0.device) * f(xa).to(x0.device) > 0
        xa[low] = x1[low]
        xb[~low] = x1[~low]

        x0 = x1

    print(f"newton not converged")
    return x0


def bisect(
    f: Callable,
    df: Callable,
    a: float,
    b: float,
    x0: Optional[Tensor] = None,
    max_iter: int = ITER,
):
    del df
    if torch.any(f(a) * f(b) > 0):
        raise ValueError(f"None or no unique root in given intervall [{a},{b}]")

    # Define lower/upper boundaries as tensor
    xa = a * torch.ones_like(x0)
    xb = b * torch.ones_like(x0)

    for i in range(max_iter):
        # define midpoint
        x1 = (xa + xb) / 2

        # adjusting brackets
        higher = f(x1) > 0
        xa[~higher] = x1[~higher]
        xb[higher] = x1[higher]

        if torch.allclose(xa, xb, atol=XTOL, rtol=RTOL):
            # print(f"coverged after {i+1} iterations")
            return x1

        x0 = x1

    print(f"bisect not converged")
    return x0
