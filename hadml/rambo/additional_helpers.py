import torch


import torch

def vectorized_boost(momenta_4, total_4_momenta, inverse=False):
    k = momenta_4
    P = total_4_momenta

    E = P[:, 0]
    p_vec = P[:, 1:]

    beta = p_vec / E.unsqueeze(1)              # (B,3)
    beta2 = (beta ** 2).sum(dim=1)             # (B,)
    gamma = 1.0 / torch.sqrt(1 - beta2)        # (B,)

    if inverse:
        beta = -beta

    beta  = beta[:, None, :]                   # (B,1,3)
    gamma = gamma[:, None, None]               # (B,1,1)

    # Stable expression: (γ−1)/β² = 1/(1+γ)
    gm1_over_beta2 = 1.0 / (1.0 + gamma)       # (B,1,1)

    k0   = k[:, :, 0:1]
    kvec = k[:, :, 1:]

    beta_dot_k = (beta * kvec).sum(dim=2, keepdim=True)

    k0_new = gamma * (k0 - beta_dot_k)

    kvec_new = (
        kvec
        + gm1_over_beta2 * beta_dot_k * beta   # uses stable formula
        - gamma * k0 * beta
    )

    return torch.cat([k0_new, kvec_new], dim=2)




def lorentz_boost(hadrons_rest_frame, cluster, inverse=False):
    eps = 1e-6
    E_clamped = cluster[0].clamp(min=eps)
    beta_vec = cluster[1:] / E_clamped
    beta_mag_sq = torch.dot(beta_vec, beta_vec)
    beta_mag_sq_clamped = torch.clamp(beta_mag_sq, max=1.0 - eps)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - beta_mag_sq_clamped, min=eps))

    transformed_hadrons = []
    for hadron in hadrons_rest_frame:
        Eh, pxh, pyh, pzh = hadron
        if beta_mag_sq_clamped < eps:
            transformed_hadrons.append(hadron)
            continue

        p_h = torch.tensor([pxh, pyh, pzh], dtype=cluster.dtype, device=cluster.device)
        beta_dot_p = torch.dot(beta_vec, p_h)

        if not inverse:
            E_prime = gamma * (Eh - beta_dot_p)
            factor = (gamma - 1.0) * beta_dot_p / beta_mag_sq_clamped - gamma * Eh
        else:
            E_prime = gamma * (Eh + beta_dot_p)
            factor = (gamma - 1.0) * beta_dot_p / beta_mag_sq_clamped + gamma * Eh

        p_prime = p_h + factor * beta_vec
        transformed_hadrons.append(torch.cat([E_prime.unsqueeze(0), p_prime]))

    return torch.stack(transformed_hadrons)


def get_invariant_mass(four_momentum):
    E = four_momentum[..., 0]
    momentum_sq = four_momentum[..., 1:].pow(2).sum(dim=-1)
    mass_sq = E.pow(2) - momentum_sq
    return torch.sqrt(torch.clamp(mass_sq, min=0.0))