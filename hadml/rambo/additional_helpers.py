import torch


def lorentz_boost(hadrons_rest_frame, cluster, inverse=False):
    E = cluster[0]
    p_vec = cluster[1:]

    beta_vec = p_vec / E
    beta_mag = torch.norm(beta_vec)
    gamma = 1.0 / torch.sqrt(1 - beta_mag**2)

    transformed_hadrons = []

    for hadron in hadrons_rest_frame:
        Eh, pxh, pyh, pzh = hadron
        p_h = torch.tensor([pxh, pyh, pzh], dtype=cluster.dtype, device=cluster.device)

        beta_dot_p = torch.dot(beta_vec, p_h)

        if not inverse:
            E_prime = gamma * (Eh - beta_dot_p)
            p_prime = p_h + ((gamma - 1) * beta_dot_p / beta_mag**2 - gamma * Eh) * beta_vec
        else:
            E_prime = gamma * (Eh + beta_dot_p)
            p_prime = p_h + ((gamma - 1) * beta_dot_p / beta_mag**2 + gamma * Eh) * beta_vec

        transformed_hadrons.append(torch.cat([E_prime.unsqueeze(0), p_prime]))

    return torch.stack(transformed_hadrons)