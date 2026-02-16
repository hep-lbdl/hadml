import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List, Iterable, Optional, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback
try:
    from pytorch_lightning.loggers import LightningLoggerBase
except ImportError:
    try:
        from pytorch_lightning.loggers.base import LightningLoggerBase
    except ImportError:
        from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, Tensor
from hadml.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig, **kwargs):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg, **kwargs)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = (
                f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            )
            save_file(
                path, content
            )  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def get_wasserstein_grad_penalty(
    D: nn.Module,
    real_inputs: Union[Iterable[torch.Tensor], torch.Tensor],
    fake_inputs: Union[Iterable[torch.Tensor], torch.Tensor],
):
    """Gradient penalty from https://arxiv.org/abs/1704.00028"""
    if isinstance(real_inputs, torch.Tensor):
        real_inputs = [real_inputs]
    if isinstance(fake_inputs, torch.Tensor):
        fake_inputs = [fake_inputs]
    if (len(real_inputs) != len(fake_inputs)) or np.any(
        [real.shape != fake.shape for real, fake in zip(real_inputs, fake_inputs)]
    ):
        raise ValueError("Inputs must match in length and shapes!")

    device = real_inputs[0].device
    alphas = [torch.rand(x.shape[0], 1).to(device) for x in real_inputs]

    interpolates = [
        alpha * real + ((1 - alpha) * fake)
        for alpha, real, fake in zip(alphas, real_inputs, fake_inputs)
    ]
    interpolates = [x.requires_grad_(True) for x in interpolates]
    score = D(*interpolates)

    gradients = torch.autograd.grad(
        outputs=score.sum(), inputs=interpolates, create_graph=True, retain_graph=True
    )
    gradients = torch.cat(gradients, dim=1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_r1_grad_penalty(
    D: nn.Module, real_inputs: Union[Iterable[torch.Tensor], torch.Tensor]
):
    """Gradient penalty from https://arxiv.org/abs/1801.04406"""
    if isinstance(real_inputs, torch.Tensor):
        real_inputs = [real_inputs]

    real_inputs = [x.requires_grad_(True) for x in real_inputs]
    score = D(*real_inputs)

    gradients = torch.autograd.grad(
        outputs=score.sum(), inputs=real_inputs, create_graph=True, retain_graph=True
    )
    gradients = torch.cat(gradients, dim=-1)

    if gradients.dim() >= 3:
        gradient_penalty = gradients.norm(2, dim=[-1, -2]).pow(2).mean() 
    else:
        gradient_penalty = gradients.pow(2).sum(-1).mean()

    return gradient_penalty

def get_r1_grad_penalty_2(
    D: nn.Module,
    real_inputs: torch.Tensor,
    real_inputs_rem: torch.Tensor
):
    """
    R1 gradient penalty ONLY wrt real_inputs.
    """

    # R1 must differentiate only wrt data
    real_inputs = real_inputs.requires_grad_(True)

    # critical: prevent gradient path
    real_inputs_rem = real_inputs_rem.detach()

    # discriminator forward
    score, _ = D(real_inputs, real_inputs_rem)

    # sum over batch to get scalar
    grad = torch.autograd.grad(
        outputs=score.sum(),
        inputs=real_inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # standard R1 penalty
    grad = grad.view(grad.size(0), -1)
    gradient_penalty = (grad.pow(2).sum(1)).mean()

    return gradient_penalty







def conditional_cat(optional: Optional[Tensor], x: Tensor, dim=1):
    if optional is None:
        return x
    return torch.cat([optional, x], dim=dim)


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


import torch



def safe_atan2(y, x, eps=1e-12):
    """
    Safe atan2 with custom gradient to avoid NaN when x=y=0
    """
    # Forward pass: normal atan2
    result = torch.atan2(y, x)
    
    # For backward pass, we need to handle x=0, y=0 case
    # We'll use a custom backward function
    class SafeAtan2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, x):
            ctx.save_for_backward(y, x)
            return torch.atan2(y, x)
        
        @staticmethod
        def backward(ctx, grad_output):
            y, x = ctx.saved_tensors
            # Normal gradient for atan2: d/dx = -y/(x^2 + y^2)
            #                      d/dy = x/(x^2 + y^2)
            denom = x**2 + y**2 + eps
            grad_x = -y / denom * grad_output
            grad_y = x / denom * grad_output
            
            # When both x and y are near zero, set gradient to 0
            small_mask = (x.abs() < eps) & (y.abs() < eps)
            if small_mask.any():
                grad_x = torch.where(small_mask, torch.zeros_like(grad_x), grad_x)
                grad_y = torch.where(small_mask, torch.zeros_like(grad_y), grad_y)
                
            return grad_y, grad_x
    
    return SafeAtan2.apply(y, x)


def lorentz_to_kt_eta_phi_m(p: torch.Tensor, eps: float = 1e-12):
    E  = p[..., 0]
    px = p[..., 1]
    py = p[..., 2]
    pz = p[..., 3]

    # ---- transverse momentum ----
    pt = torch.sqrt(px**2 + py**2 + eps)

    # ---- azimuth using safe_atan2 ----
    phi = safe_atan2(py, px, eps=1e-12)

    # ---- invariant mass ----
    mass2 = E**2 - (px**2 + py**2 + pz**2)
    m = torch.sqrt(torch.clamp(mass2, min=eps))

    # ---- |p| ----
    p_mag = torch.sqrt(px**2 + py**2 + pz**2 + eps)

    # ---- pseudorapidity ----
    ratio = torch.clamp(pz / (p_mag + eps), min=-0.999999, max=0.999999)
    eta = torch.atanh(ratio)

    return torch.stack((pt, eta, phi, m), dim=-1)






import os
import pickle
from particle import Particle


def pid_map(pid_map_filepath: str = None):
    """Load PID map from file and return a dictionary mapping PID to index."""
    with open(os.path.normpath(pid_map_filepath), "rb") as f:
        raw_pid_map = pickle.load(f)             # {pid: index}


    pid_map = {idx: (0.0 if pid == 'uncommon_pid' else float(pid))
            for pid, idx in raw_pid_map.items()}
    # check if uncommon_pid is in pid_map
    find_pid = False
    for pid in pid_map.values():
        if pid == 0.0:
            find_pid = True
    print('----------------------------')
    print('----------------------------')
    print('Uncommon PID found in pid_map:', find_pid)

    # Masses in GeV.  (Negative PDG IDs = antiparticles -> same mass.)
    manual_mass_map = {5212.0: 5.8112082, 5214.0: 5.8325324, 
                        5314.0: 5.967868, 5322.0: 5.897625, 
                        10511.0: 5.726344, 10513.0: 5.7207456, 
                        10521.0: 5.726035, 10523.0: 5.720276, 
                        10531.0: 5.8176956, 10533.0: 5.8293395, 
                        13322.0: 1.689997, 15122.0: 5.912, 15322.0: 6.1110806, 
                        20413.0: 2.4376314, 20513.0: 5.7615266, 
                        20523.0: 5.762014, 20533.0: 5.829, 
                        100311.0: 1.4600005, 100321.0: 1.4595373,
                        545: 7.35, 5312: 5.96, 5324: 5.97, 5334: 6.13,
                        10541: 7.25, 10543: 7.3, 
                        13312: 1.69, 14312: 2.79, 14322: 2.79, 23312: 1.96, 23322: 1.96,
                        15312.0: 6.11}
    # manual_mass_map = {5212.0: 5.8112082}
    # what is the mass of 15312 ? 


    masses = []
    not_found = []

    for pid in pid_map.values():
        if pid == 0.0:
            masses.append(0.0)
            continue

        pid_int = abs(float(pid))
        
        # Check manual overrides first
        if pid_int in manual_mass_map:
            masses.append(manual_mass_map[pid_int])
            continue
        
        pid_int = int(pid)
        # Otherwise try PDG lookup
        try:
            m = Particle.from_pdgid(pid_int).mass / 1000  # -> GeV
            masses.append(m)
        except Exception:
            masses.append(0.0)
            not_found.append(pid_int)

    # Print summary of missing masses
    if not_found:
        print("masses not found for PIDs:", sorted(set(not_found)))
        for pid_u in sorted(set(not_found)):
            print("mass not found, set to 0 for PID:", pid_u)

    return masses


