import pyrootutils
import wandb

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
OmegaConf.register_new_resolver("gen_list", lambda x, y: [x] * y)

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from hadml import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig, logger=None) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    if not logger:
        log.info("Instantiating loggers...")
        logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def sweep(cfg: DictConfig) -> Optional[float]:
    """Hyperparameter sweep example."""
    init_sweep = not cfg.get("sweep_id")

    if init_sweep:
        sweep_configuration = {
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "val/swd"},
            "parameters": {
                "r1_reg": {"max": 5000,
                           "min": 1,
                           'distribution': 'log_uniform_values'},
                "lr": {"max": 0.005,
                       "min": 0.00001,
                       'distribution': 'log_uniform_values'},
                "width":  {"values": [50, 100, 250, 500]},
                "depth": {"max": 8, "min": 4},
                "batch_size": {"values": [5000, 10000, 20000, 40000, 80000]},
                "batch_norm_gen": {"values": [1, 2, 3]},
                # "batch_norm_dis": {"values": [0, 1, 2]},
                "noise_dim": {"values": [1, 16, 64]},
                "num_critics": {"values": [1, 2, 3]},
                "num_gen": {"values": [1, 2]},
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration,
                               entity=cfg.logger.wandb.entity,
                              project=cfg.logger.wandb.project)
        print(f"Sweep id: {sweep_id}")
    else:
        wandb.agent(cfg.sweep_id,
                    function=lambda: train_wandb(cfg),
                    count=2,
                    entity=cfg.logger.wandb.entity,
                    project=cfg.logger.wandb.project
                    )

    return None

def train_wandb(cfg: DictConfig) -> None:
    """Hyperparameter sweep example."""
    logger = utils.instantiate_loggers(cfg.get("logger"))

    cfg.model.r1_reg = wandb.config.r1_reg
    cfg.model.optimizer_generator.lr = wandb.config.lr
    cfg.model.optimizer_discriminator.lr = wandb.config.lr
    cfg.model.generator.hidden_dims = wandb.config.depth * [wandb.config.width]
    cfg.model.discriminator.hidden_dims = wandb.config.depth * [wandb.config.width]
    cfg.model.num_critics = wandb.config.num_critics
    cfg.model.num_gen = wandb.config.num_gen
    cfg.datamodule.batch_size = wandb.config.batch_size
    cfg.model.noise_dim = wandb.config.noise_dim
    cfg.model.generator.batch_norm = wandb.config.batch_norm_gen
    # cfg.model.discriminator.batch_norm = wandb.config.batch_norm_dis
    train(cfg, logger=logger)


if __name__ == "__main__":
    main()
    # sweep()
