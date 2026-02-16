import torch

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
    xi = torch.zeros_like(M) # output

    if not valid.any():
        return xi

    # work only on valid events
    p_v   = p[valid]
    p2_v = p2[valid]
    m_v   = m[valid]
    m2_v = m2[valid]
    M_v   = M[valid]
    mask_v = mask[valid]

    denom = (p_v.abs() * mask_v).sum(dim=1)
    xi_v = (M_v / denom).clamp_min(1e-8)
   # print(f'xi_v: {xi_v}')
    # Newton iterations
    for _ in range(n_iter):
        # inside sqrt
        inside = xi_v[:,None]**2 * p2_v + m2_v
        inside = inside.clamp_min(eps)

        # masked energies
        E_i = torch.sqrt(inside) * mask_v

        # f = ΣE_i - M
        f = E_i.sum(dim=1) - M_v
        print(f'f: {f}')

        # df/dxi = Σ xi*p^2 / (E_i)   (masked)
        dE_dxi = (xi_v[:,None] * p2_v / E_i.clamp_min(eps)) * mask_v
        f_prime = dE_dxi.sum(dim=1).clamp_min(1e-10)

        xi_v = (xi_v - f/f_prime).clamp_min(0.0)

    # insert results back
    xi[valid] = xi_v
    return xi


# ----------------------------
# parameters for test
# ----------------------------
import torch, time

# ---------------- SETTINGS ----------------
B = 256
N = 30
n_iter = 10                     # Newton iterations (differentiable)
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
p = np.load('debug_momenta_mag_rest_frame.npy')[:2]
m = np.load('debug_masses.npy')[:2]
M = np.load('debug_cluster_invariant_masses.npy')[:2]
mask = ~np.load('debug_pad_mask.npy')[:2]

# invert the mask to match the expected format
#mask = np.where(mask == 1, 0, 1)
#print('mask:', mask)
#print('mask shape:', mask)

print('Input p:', p[0])
print('Input m:', m[0])
print('Input M:', M[0])

p = torch.tensor(p, device=device, dtype=dtype).requires_grad_(True)
m = torch.tensor(m, device=device, dtype=dtype).requires_grad_(True)
M = torch.tensor(M, device=device, dtype=dtype).requires_grad_(True)
mask = torch.tensor(mask, device=device, dtype=torch.bool).requires_grad_(False)
#print('Input mask:', mask[0])
print('Input mask:', mask)
xi = solve_xi(p, m, M, mask, n_iter=n_iter)
print('Computed xi:', xi)

# torch.manual_seed(0)

# # random parameters
# p = (torch.rand(B, N, device=device, dtype=dtype) * 3.0).requires_grad_(True)
# m = (torch.rand(B, N, device=device, dtype=dtype) * 0.4 + 0.1).requires_grad_(True)
# M = (m.sum(dim=1) + 5.0).to(device)    # ensure kinematically allowed
# M.requires_grad_(True)                 # also test gradient wrt M

# # warmup run (GPU JIT, cache)
# # _ = solve_xi(p, m, M, n_iter=n_iter)

# # ---------------- TIMING FORWARD ----------------
# t0 = time.time()
# xi = solve_xi(p, m, M, n_iter=n_iter)
# print("Computed xi:", xi)
# torch.cuda.synchronize() if device=="cuda" else None
# t1 = time.time()

# # ---------------- TIMING BACKWARD ----------------
# loss = xi.sum()               # simple scalar to backprop
# loss.backward()
# torch.cuda.synchronize() if device=="cuda" else None
# t2 = time.time()

# print("\n--- TIMING ---")
# print(f"Batch size: {B}, N={N}, Newton iterations={n_iter}")
# print(f"Forward pass time:  {(t1-t0)*1000:.3f} ms")
# print(f"Backward pass time: {(t2-t1)*1000:.3f} ms")
# print(f"Total (fwd+back):   {(t2-t0)*1000:.3f} ms")

# print("\nGradient sanity check:")
# print("dξ/dp:", p.grad.mean().item())
# print("dξ/dm:", m.grad.mean().item())
# print("dξ/dM:", M.grad.mean().item())

