# Code from https://github.com/madgraph-ml/madspace

from typing import Optional, Tuple
from math import gamma, pi
import torch
from torch import Tensor, sqrt, atan2
from hadml.rambo.roots import get_u_parameter, get_xi_parameter
from hadml.rambo.base import PhaseSpaceMapping, TensorList
from hadml.rambo.kinematics import boost, mass, esquare
from hadml.rambo.ps_utils import build_p_in, map_fourvector_rambo_diet, two_body_decay_factor


class NearlyRambo():
    """Rambo on Diet algorithm as presented in
        [2] Rambo on diet - https://arxiv.org/abs/1308.2922

    Note, that here is an error in the algorithm of [2], which has been fixed.
    For details see:
        [3] RW's PhD thesis - https://doi.org/10.11588/heidok.00029154
        [4] ELSA paper - https://arxiv.org/abs/2305.07696
    """

    # def __init__(
    #     self,
    #     nparticles: int,
    #     masses: list[float] = None,
    #     check_emin: bool = False,
    # ):


    def map(self, inputs: TensorList, nparticles=2, masses=None):

        self.nparticles = nparticles
        self.check_emin = False

       # self.check_emin = check_emin

        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.nparticles
            self.e_min = sum(masses)
        else:
            self.masses = masses
            self.e_min = 0.0

        dims_in = [(3 * nparticles - 4,), ()]
        dims_out = [(nparticles, 4)]

        self.pi_factors = (2 * pi) ** (4 - 3 * self.nparticles)

       # super().__init__(dims_in, dims_out)
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm]
                r: random numbers with shape=(b,3*n-4)
                e_cm: COM energy with shape=(b,)

        Returns:
            p_ext (Tensor): external momenta with shape=(b,n+2,4)
            det (Tensor): det of mapping with shape=(b,)
        """
       # del condition
        r = inputs[0]  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()

        # Check if the partonic COM energy is large enough
        with torch.no_grad():
            if self.check_emin:
                if torch.any(e_cm <= self.e_min):
                    raise ValueError(
                        f"partonic COM energy needs to be larger than sum of external masses!"
                    )

        # Do rambo on diet loop
        # prepare momenta
        p_out = torch.empty((r.shape[0], self.nparticles, 4), device=r.device)

        print('r shape:', r.shape)
        # split random numbers in energy and angular
        ru, romega = r[:, : self.nparticles - 2], r[:, self.nparticles - 2 :]
        print('ru shape:', ru.shape)
        # Solve rambo equation numerically for all u directly
        # u has shape=(b, nparticles - 2)
        a_all = torch.tensor([2., 2.71200135, 3.08452063, 3.31307332, 3.46751197, 3.57883263, 3.66287206, 3.72856063, 3.78131568])
        a = a_all[:self.nparticles - 2].flip(0)
       # print('a shape:', a.shape)
        m = torch.arange(self.nparticles - 2, 0, -1)
        b = a - ru * (a - 2)
        x = 2 * ru / (b + torch.sqrt(b*b + 4*ru*(1-b)))
        u = x ** (1 / (2*m))
        xr = 1 - x
        #jac_x = (2 * x * xr + a * xr * xr) / (1 + (a - 2) * x * xr) ** 2
        #jac = torch.prod((1 - u * u) / jac_x, dim=1)
        
        # split into rcos and rphi
        rcos, rphi = romega.split(self.nparticles - 1, dim=1)

        # Define all angles vectorized
        cos_theta = 2 * rcos - 1
        phi = 2 * pi * rphi

        # Define intermediate masses
        if self.masses is None:
            e_cm_massless = e_cm
            M = torch.zeros((r.shape[0], self.nparticles), device=r.device)
            M[:, 0] = e_cm
            M[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm[:, None]
           # w_m = 1
            # Define first n-1 energies
            # gets shape (b, nparticles - 1)
            q = 4 * M[:, :-1] * two_body_decay_factor(M[:, :-1], M[:, 1:], 0)
            pnm1 = map_fourvector_rambo_diet(q, cos_theta, phi)
        else:
            masses_cum = self.masses.flip(0).cumsum(dim=0).flip(0)
            e_cm_massless = e_cm - masses_cum[0]
            K = torch.zeros((r.shape[0], self.nparticles), device=r.device)
            K[:, 0] = e_cm_massless
            K[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm_massless[:, None]
            M = K + masses_cum
            #rho_K = two_body_decay_factor(K[:, :-1], K[:, 1:], 0)
            rho_M = two_body_decay_factor(M[:, :-1], M[:, 1:], self.masses[:-1])
          #  w_m = torch.prod(rho_M / rho_K, dim=1) * torch.prod(M[:, 1:-1] / K[:, 1:-1], dim=1) #/ 8
            q = 4 * M[:, :-1] * rho_M
            pnm1 = map_fourvector_rambo_diet(q, cos_theta, phi)
            pnm1[:, :, 0] = torch.sqrt(pnm1[:, :, 0]**2 + self.masses[:-1]**2)

        # Define Qs
        Q = e_cm[:, None] * torch.tile(torch.tensor([1, 0, 0, 0]), (r.shape[0], 1))

        # Define loop over (n-1 particles) boosts
        for i in range(self.nparticles - 1):
            # Define Qi
            Q0_i = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
            Qp_i = -pnm1[:, i, 1:]
            Q_i = torch.concat([Q0_i[:, None], Qp_i], dim=1)

            # Boost p_i and Q_i along Q
            p_out[:, i] = boost(pnm1[:, i], Q)
            Q = boost(Q_i, Q)

        # Define final particle
        p_out[:, self.nparticles - 1] = Q

        # Get massless phase-space weights
        torch_ones = torch.ones((r.shape[0],), device=r.device)
      #  w0 = torch_ones * self._massles_weight(e_cm_massless) * jac

        # construct initial state momenta
        p_in = build_p_in(e_cm)

        p_ext = torch.cat([p_in, p_out], dim=1)
        return (p_ext,)






class RamboOnDiet():
    """Rambo on Diet algorithm as presented in
        [2] Rambo on diet - https://arxiv.org/abs/1308.2922

    Note, that here is an error in the algorithm of [2], which has been fixed.
    For details see:
        [3] RW's PhD thesis - https://doi.org/10.11588/heidok.00029154
        [4] ELSA paper - https://arxiv.org/abs/2305.07696
    """

    def map(self, inputs: TensorList, nparticles=2, masses=None):
        """Map from random numbers to momenta"""
        r = inputs[0]  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()

        # Do rambo on diet loop
        # prepare momenta
        p_out = torch.empty((r.shape[0], nparticles, 4), device=r.device)

        # split random numbers in energy and angular
        ru, romega = r[:, : nparticles - 2], r[:, nparticles - 2 :]

        # Solve rambo equation numerically for all u directly
        # u has shape=(b, nparticles - 2)
        u = get_u_parameter(ru)

        # split into rcos and rphi
        rcos, rphi = romega.split(nparticles - 1, dim=1)

        # Clamp angular inputs to avoid costheta = +/-1 or phi out of [0, 2pi)
        ang_eps = 1e-6
        rcos = rcos.clamp(ang_eps, 1 - ang_eps)
        rphi = rphi.clamp(ang_eps, 1 - ang_eps)

        # Define all angles vectorized
        cos_theta = 2 * rcos - 1
        phi = 2 * pi * rphi

        # Define intermediate masses
        M = torch.zeros((r.shape[0], nparticles), device=r.device)
        M[:, 0] = e_cm
        M[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm[:, None]

        # Define first n-1 energies
        # gets shape (b, nparticles - 1)
        q = 4 * M[:, :-1] * two_body_decay_factor(M[:, :-1], M[:, 1:], 0)

        # Define first (n-1) particles
        pnm1 = map_fourvector_rambo_diet(q, cos_theta, phi)

        # Define Qs
        Q = e_cm[:, None] * torch.tile(torch.tensor([1, 0, 0, 0]), (r.shape[0], 1))

        # Define loop over (n-1 particles) boosts
        for i in range(nparticles - 1):
            # Define Qi
            Q0_i = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
            Qp_i = -pnm1[:, i, 1:]
            Q_i = torch.concat([Q0_i[:, None], Qp_i], dim=1)

            # Boost p_i and Q_i along Q
            p_out[:, i] = boost(pnm1[:, i], Q)
            Q = boost(Q_i, Q)

        # Define final particle
        p_out[:, nparticles - 1] = Q

        # Get massless phase-space weights
        torch_ones = torch.ones((r.shape[0],), device=r.device)
        w0 = torch_ones * self._massles_weight(e_cm, nparticles)

        # construct initial state momenta
        p_in = build_p_in(e_cm)

        if masses is not None:
            # match dimensions of masses
            m = masses[None, ...].to(device=r.device)

            # solve for xi in massive case, see Ref. [1]
            xi = get_xi_parameter(p_out[:, :, 0], m)

            # Make momenta massive
            xi = xi[:, None, None]
            k_out = torch.empty_like(p_out)
            k_out[:, :, 0] = sqrt(m**2 + xi[:, :, 0] ** 2 * p_out[:, :, 0] ** 2)
            k_out[:, :, 1:] = xi * p_out[:, :, 1:]
            # Get massive density corr. factor
            w_m = self._massive_weight(nparticles, k_out, p_out, xi[:, 0, 0])

            p_ext = torch.cat([p_in, k_out], dim=1)
            return (p_ext,), w_m * w0

        p_ext = torch.cat([p_in, p_out], dim=1)
        return (p_ext,), w0

    def map_inverse(self, inputs: TensorList, nparticles=2, masses=None):
        """Map from momenta to random numbers"""
        # Get input momenta
        p_ext = inputs[0]
        k = p_ext[:, 2:]
        e_cm = (k.sum(dim=1))[:, 0]
        w0 = self._massles_weight(e_cm, nparticles)

        # Make momenta massless before going back to random numbers
        p = torch.empty((k.shape[0], nparticles, 4), device=p_ext.device)
        if masses is not None:
            # Define masses
            m = masses[None, ...]
            # solve for xi in massive case, see Ref. [1], here analytic result possible!
            xi = torch.sum(sqrt(k[:, :, 0] ** 2 - m**2), dim=-1) / e_cm

            # Make them massless
            xi = xi[:, None, None]
            p[:, :, 0] = torch.sqrt(k[:, :, 0] ** 2 - m**2) / xi[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:] / xi
            wm = self._massive_weight(nparticles, k, p, xi[:, 0, 0])
        else:
            xi = None
            wm = 1.0
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]

        # Get random numbers associated to the intermediate masses
        # have shapees (b, n-1)
        P = torch.cumsum(p.flip(1), dim=1)[:, 1:]  # has shape (b, n-1)
        M = mass(P)
        # have shapes (b, n-2)
        um = (M[:, :-1] / M[:, 1:]).flip(1)
        iarray = torch.arange(2, nparticles, device=p.device)[None, :]
        uc = nparticles + 1 - iarray
        uexp = 2 * (nparticles - iarray)
        ru = uc * um**uexp - (uc - 1) * um ** (uexp + 2)

        # Get the angles in correct frames
        # Here: do all boosts in one go
        Q = P.flip(1)
        p_prime = boost(p[:, :-1], Q, inverse=True)
        pmag = sqrt(esquare(p_prime[..., 1:]))
        costheta = p_prime[..., 3] / pmag
        phi = atan2(p_prime[..., 2], p_prime[..., 1])

        # Define the random numbers
        rcos = 0.5 * (costheta + 1.0)
        rphi = phi / (2 * pi) + (phi < 0)

        # Concat angular and energy random numbers
        r = torch.cat([ru, rcos, rphi], dim=-1)
        return (r, e_cm), 1.0 / w0 / wm

    def _massles_weight(self, e_cm, nparticles):
        w0 = (
            (pi / 2.0) ** (nparticles - 1)
            * e_cm ** (2 * nparticles - 4)
            / (gamma(nparticles) * gamma(nparticles - 1))
        )
        return w0 * (2 * pi) ** (4 - 3 * nparticles)

    def _massive_weight(
        self,
        nparticles: int,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            nparticles (int): number of particles
            k (torch.Tensor): massive momenta in shape=(b,n,4)
            p (torch.Tensor): massless momenta in shape=(b,n,4)
            xi (torch.Tensor, Optional): shift variable with shape=(b,)

        Returns:
            torch.Tensor: massive weight
        """
        # get correction factor for massive ones
        ks2 = k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2
        ps2 = p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2
        k0 = k[:, :, 0]
        p0 = p[:, :, 0]
        w_M = (
            xi ** (3 * nparticles - 3)
            * torch.prod(p0 / k0, dim=1)
            * torch.sum(ps2 / p0, dim=1)
            / torch.sum(ks2 / k0, dim=1)
        )
        return w_M

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs