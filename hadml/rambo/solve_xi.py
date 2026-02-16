import torch
import sys

def solve_xi(p, m, M, mask, n_iter=10, eps=1e-12):
    """
    Solve xi from:
        sum_i mask_i * sqrt(xi^2 * p_i^2 + m_i^2) = M

    p      : (B,N)  momenta magnitudes
    m      : (B,N)  masses
    M      : (B,)   target invariant mass
    mask   : (B,N)  boolean or {0,1}, 1 = real particle, 0 = padded
    returns: xi (B,), xi=0 for kinematically impossible events
    """


    p2 = p*p
    m2 = m*m

    # event-level kinematic validity
    mass_sum = (m * mask).sum(dim=1)
    valid = mass_sum <= M    # (B,)
    # relu on valid 
    violation = torch.nn.functional.relu(-M + mass_sum)
    #print('violation:', violation.shape)
    # mass > M


    xi = torch.zeros_like(M) # output

    if not valid.any():
        return xi, valid, violation

    # work only on valid events
    p_v   = p[valid]
    p2_v = p2[valid]
    m_v   = m[valid]
    m2_v = m2[valid]
    M_v   = M[valid]
    mask_v = mask[valid]

    denom = (p_v.abs() * mask_v).sum(dim=1).clamp_min(1e-8)
    xi_v = (M_v / denom).clamp_min(1e-8)
   # print(f'xi_v: {xi_v}')
    # Newton iterations
    for _ in range(n_iter):
        # inside sqrt
        inside = xi_v[:,None]**2 * p2_v + m2_v

        # masked energies
        E_i = torch.sqrt(torch.clamp(inside, min=eps)) * mask_v
        # f = ΣE_i - M
        f = E_i.sum(dim=1) - M_v

        # df/dxi = Σ xi*p^2 / (E_i)   (masked)
        dE_dxi = (xi_v[:,None] * p2_v / E_i.clamp_min(eps)) * mask_v
        f_prime = dE_dxi.sum(dim=1).clamp_min(1e-8)

        xi_v = (xi_v - f/f_prime).clamp_min(0.0)

    # insert results back
    xi[valid] = xi_v

    #print('xi:', xi.shape)
    #print('valid:', valid.shape)
   # sys.exit()
    return xi, valid, violation
