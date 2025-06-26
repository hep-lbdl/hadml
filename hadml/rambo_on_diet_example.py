import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hadml.rambo.rambo_on_diet import RamboOnDiet
from hadml.rambo.additional_helpers import lorentz_boost
import torch


def rambo_on_diet_example(
    nparticles=2,
    masses=[0.13765123, 0.13954307],
    E_CM=2.443165,
    cluster=torch.tensor([31.1089, 11.0606, 23.0724, -17.5249]),
    p=torch.tensor([[
        [0, 0, 0, 0], # template to comply with the interface 
        [0, 0, 0, 0], # template to comply with the interface
        [27.6396, 9.26599, 20.9831, -15.4201], 
        [3.46926, 1.79465, 2.08936, -2.10478]]]),
):
    """
    Example of using RamboOnDiet to map momenta in a decay process.
    This example uses a specific decay process with two final state particles.
    """
    
    # Rambo initialisation
    rambo = RamboOnDiet(nparticles=nparticles, masses=masses)

    # Original momenta with two zeroed particles (initial state) and two non-zeroed particles (final state)    
    print('\n', '*' * 20, sep='')
    print("Input momenta shape:", p.shape)
    print("Input momenta:\n", p)
    print("Sum of momenta:", p.sum(dim=1))

    # Mapping the momenta to the phase space
    (r, e_cm), det = rambo.map_inverse(inputs=[p])
    print("Inverse mapped momenta shape:", r.shape)
    print("Inverse mapped momenta:\n", r)
    print("Inverse e_cm (in the COM frame would be equal to the invariant mass):", e_cm)
    print("Determinant:", det)

    # Getting the mapped momenta back to the original space
    (p,), weight = rambo.map([r, torch.tensor([E_CM])])
    print("Mapped momenta shape:", p.shape)
    print("Mapped momenta:\n", p[:, 2:])  # Only the final state particles
    print("Weight:", weight)
    print("Sum of momenta:", p[:, 2:].sum(dim=1))

    # Applying the inverse Lorentz transformation to the mapped momenta
    p_lab_frame = lorentz_boost(p[0, 2:], cluster, inverse=True)
    print("Hadrons in the lab frame:\n", p_lab_frame)
    print("Sum of momenta in the lab frame:", p_lab_frame.sum(dim=0))
    return p_lab_frame


# Sample 1
rambo_on_diet_example()

# Sample 2
rambo_on_diet_example(
    nparticles=6,
    masses=[0.13957083, 0.13957106, 0.13497813, 0.13960445, 0.13960269, 0.13497782],
    E_CM=6.5039706,
    cluster=torch.tensor([7.37021, -1.42063, -3.03948, 0.872776]),
    p=torch.tensor([[
        [0, 0, 0, 0],  # template to comply with the interface
        [0, 0, 0, 0],  # template to comply with the interface
        [0.484152, -0.10366, 0.445061, 0.0780926],
        [0.720177, -0.338852, -0.438608, 0.438152],
        [0.752091, 0.304669, -0.534168, 0.411416],
        [3.0213, -0.278835, -2.72083, -1.27597],
        [1.59697, -0.848362, 0.934673, 0.968242],
        [0.795524, -0.155593, -0.725604, 0.252842]
    ]])
)