import torch


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