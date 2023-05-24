from typing import Any, List, Optional, Dict, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torch.optim import Optimizer
import ot

from hadml.metrics.media_logger import log_images
from hadml.utils.utils import (
    get_wasserstein_grad_penalty,
    conditional_cat,
    get_r1_grad_penalty,
    get_one_hot,
)


class CondParticleGANModule(LightningModule):
    """Conditional GAN predicting particle momenta and types.
    Parameters:
        noise_dim: dimension of noise vector
        num_particle_ids: maximum number of particle types
        num_output_particles: number of outgoing particles
        num_particle_kinematics: number of outgoing particles' kinematic variables
        generator: generator network
        discriminator: discriminator network
        loss_type: type of loss function to use, ['bce', 'wasserstein', 'ls']
        optimizer_generator: generator optimizer
        optimizer_discriminator: discriminator optimizer
        comparison_fn: function to compare generated and real data
    """

    def __init__(
        self,
        noise_dim: int,
        cond_info_dim: int,
        num_particle_ids: int,
        num_output_particles: int,
        num_particle_kinematics: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: Optimizer,
        optimizer_discriminator: Optimizer,
        num_critics: int,
        num_gen: int,
        embedding_module: Optional[torch.nn.Module] = None,
        scheduler_generator: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_discriminator: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_type: str = "bce",
        wasserstein_reg: float = 0.0,
        r1_reg: float = 0.0,
        target_gumbel_temp: float = 0.3,
        comparison_fn: Optional[Callable] = None,
        save_only_improved_plots: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["generator", "discriminator", "comparison_fn", "criterion"],
        )

        self.embedding_module = embedding_module
        self.generator = generator
        # self.discriminator = discriminator
        class D(LightningModule):
            def __init__(self):
                super().__init__()
                self.d = discriminator

            def forward(self, x_generated, particle_type_data):
                particle_type_data = particle_type_data

                # return self.d(x_generated, particle_type_data)
                return self.d(x_generated, particle_type_data)

        self.discriminator = D()

        self.comparison_fn = comparison_fn

        # loss function
        self.criterion = torch.nn.BCELoss()
        self.wasserstein_reg = wasserstein_reg
        self.r1_reg = r1_reg
        self.current_gumbel_temp = 1.0
        self.target_gumbel_temp = target_gumbel_temp

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss_gen = MeanMetric()
        self.train_loss_disc = MeanMetric()
        self.val_swd = MeanMetric()
        self.val_particle_swd = MeanMetric()
        self.val_kinematic_swd = MeanMetric()

        # for tracking best so far
        self.val_min_avg_swd = MinMetric()
        self.val_min_avg_particle_swd = MinMetric()
        self.val_min_avg_kinematic_swd = MinMetric()

        self.test_swd = MeanMetric()
        self.test_particle_swd = MeanMetric()
        self.test_kinematic_swd = MeanMetric()
        self.test_swd_best = MinMetric()
        self.test_particle_swd_best = MinMetric()
        self.test_kinematic_swd_best = MinMetric()

        self.use_particle_mlp = False

        # check if generator is a particle MLP,
        # which produces particle kinematics and types in one go.
        # In MLP case, we need to split the output into two parts.
        for name, module in self.generator.named_modules():
            if "particle" in name:
                self.use_particle_mlp = True
                break

    def forward(
        self, noise: torch.Tensor, cond_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_fake = conditional_cat(cond_info, noise, dim=1)
        if self.use_particle_mlp:
            particle_kinematics, particle_types = self._call_mlp_particle_generator(x_fake)
        else:
            particle_kinematics, particle_types = self._call_mlp_generator(x_fake)
        particle_kinematics = torch.tanh(particle_kinematics)
        return particle_kinematics, particle_types


    def _call_mlp_particle_generator(
        self, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generator(x_fake)

    def _call_mlp_generator(
        self, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fakes = self.generator(x_fake)
        num_evts = x_fake.shape[0]

        particle_kinematics = fakes[:, : self.hparams.num_particle_kinematics]  # type: ignore
        particle_types = fakes[:, self.hparams.num_particle_kinematics :].reshape(  # type: ignore
            num_evts * self.hparams.num_output_particles, -1
        )  # type: ignore
        return particle_kinematics, particle_types

    def configure_optimizers(self):
        opt_gen = self.hparams.optimizer_generator(params=self.generator.parameters())  # type: ignore
        opt_disc = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())  # type: ignore

        # define schedulers
        if self.hparams.scheduler_generator is not None:
            sched_gen = self.hparams.scheduler_generator(optimizer=opt_gen)
            sched_disc = self.hparams.scheduler_discriminator(optimizer=opt_disc)

            return (
                {
                    "optimizer": opt_disc,
                    "lr_scheduler": {
                        "scheduler": sched_disc,
                        "monitor": "val/min_avg_swd",
                        "interval": "step",
                        "frequency": self.trainer.val_check_interval,
                        "strict": True,
                        "name": "DiscriminatorLRScheduler",
                    },
                    "frequency": self.hparams.num_critics,
                },
                {
                    "optimizer": opt_gen,
                    "lr_scheduler": {
                        "scheduler": sched_gen,
                        "monitor": "val/min_avg_swd",
                        "interval": "step",
                        "frequency": self.trainer.val_check_interval,
                        "strict": True,
                        "name": "GeneratorLRScheduler",
                    },
                    "frequency": self.hparams.num_gen,
                },
            )

        return (
            {"optimizer": opt_disc, "frequency": self.hparams.num_critics},
            {"optimizer": opt_gen, "frequency": self.hparams.num_gen},
        )

    def generate_noise(self, num_evts: int):
        return torch.randn(num_evts, self.hparams.noise_dim)  # type: ignore

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_min_avg_swd.reset()
        self.val_min_avg_particle_swd.reset()
        self.val_kinematic_swd.reset()
        self.test_swd_best.reset()
        self.test_particle_swd_best.reset()
        self.test_kinematic_swd_best.reset()

    def _generator_loss(self, score: torch.Tensor) -> torch.Tensor:
        loss_type = self.hparams.loss_type
        if loss_type == "wasserstein":
            # WGAN: https://arxiv.org/abs/1701.07875
            loss_gen = -score.mean(0).view(1)
        elif loss_type == "bce":
            # GAN: https://arxiv.org/abs/1406.2661
            loss_gen = F.binary_cross_entropy_with_logits(score, torch.ones_like(score))
        elif loss_type == "ls":
            # least squares GAN: https://arxiv.org/abs/1611.04076
            loss_gen = 0.5 * ((score - 1) ** 2).mean(0).view(1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss_gen

    def _discriminator_loss(
        self, score_real: torch.Tensor, score_fake: torch.Tensor
    ) -> torch.Tensor:
        loss_type = self.hparams.loss_type
        if loss_type == "wasserstein":
            loss_disc = score_fake.mean(0).view(1) - score_real.mean(0).view(1)
        elif loss_type == "bce":
            loss_disc = F.binary_cross_entropy_with_logits(
                score_real, torch.ones_like(score_real)
            ) + F.binary_cross_entropy_with_logits(
                score_fake, torch.zeros_like(score_fake)
            )
        elif loss_type == "ls":
            loss_disc = 0.5 * ((score_real - 1) ** 2).mean(0).view(1) + 0.5 * (
                score_fake**2
            ).mean(0).view(1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss_disc

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        self._update_gumbel_temp()

        cond_info, x_momenta, x_type_indices = batch
        x_type_data = x_type_indices
        if self.embedding_module is not None:
            x_type_data = self.embedding_module(x_type_data)

        num_evts = x_momenta.shape[0]
        device = x_momenta.device

        particle_type_data, x_generated = self._prepare_fake_batch(
            cond_info, num_evts, device
        )

        if optimizer_idx == 0:
            return self._discriminator_step(
                cond_info, particle_type_data, x_generated, x_momenta, x_type_data
            )

        if optimizer_idx == 1:
            return self._generator_step(particle_type_data, x_generated)

    def _update_gumbel_temp(self):
        progress = self.trainer.current_epoch / (self.trainer.max_epochs - 1)
        self.current_gumbel_temp = 1.0 - (1 - self.target_gumbel_temp) * progress
        self.log("gumbel", self.current_gumbel_temp)

    def _prepare_fake_batch(
        self, cond_info: Optional[torch.Tensor], num_evts: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = self.generate_noise(num_evts).to(device)

        particle_kinematics, particle_types = self(noise, cond_info)
        if self.embedding_module is None:
            raise NotImplementedError("Embedding module must be specified.")
        else:
            particle_types = particle_types.view(-1, self.hparams.num_particle_ids)
            particle_type_data = F.gumbel_softmax(particle_types,
                                                  self.current_gumbel_temp)
            particle_type_data = particle_type_data.reshape(
                particle_kinematics.shape[0], -1
            )

        x_generated = conditional_cat(cond_info, particle_kinematics, dim=1)
        return particle_type_data, x_generated

    def _generator_step(
        self, particle_type_data: torch.Tensor, x_generated: torch.Tensor
    ):
        score_fakes = self.discriminator(x_generated, particle_type_data).squeeze(-1)
        loss_gen = self._generator_loss(score_fakes)

        # update and log metrics
        self.train_loss_gen(loss_gen)
        self.log("lossG", loss_gen, prog_bar=True)
        return {"loss": loss_gen}

    def _discriminator_step(
        self,
        cond_info: Optional[torch.Tensor],
        particle_type_data: torch.Tensor,
        x_generated: torch.Tensor,
        x_momenta: torch.Tensor,
        x_type_data: torch.Tensor,
    ):
        # with real batch
        x_truth = conditional_cat(cond_info, x_momenta, dim=1)

        score_truth = self.discriminator(x_truth, x_type_data).squeeze(-1)
        # with fake batch
        score_fakes = self.discriminator(
            x_generated.detach(), particle_type_data.detach()
        ).squeeze(-1)
        loss_disc = self._discriminator_loss(score_truth, score_fakes)

        r1_grad_penalty, wasserstein_grad_penalty = self._get_grad_penalties(
            particle_type_data, x_generated, x_truth, x_type_data
        )

        self._log_metrics(loss_disc, r1_grad_penalty, wasserstein_grad_penalty)
        return {"loss": loss_disc + wasserstein_grad_penalty + r1_grad_penalty}

    def _log_metrics(self, loss_disc, r1_grad_penalty, wasserstein_grad_penalty):
        self.train_loss_disc(loss_disc)
        self.log("lossD", loss_disc, prog_bar=True)
        if self.wasserstein_reg > 0:
            self.log(
                "wasserstein_grad_penalty", wasserstein_grad_penalty, prog_bar=True
            )
        if self.r1_reg > 0:
            self.log("r1_grad_penalty", r1_grad_penalty, prog_bar=True)

    def _get_grad_penalties(
        self, particle_type_data, x_generated, x_truth, x_type_data
    ):
        wasserstein_grad_penalty = 0.0
        if self.wasserstein_reg > 0:
            wasserstein_grad_penalty = (
                get_wasserstein_grad_penalty(
                    self.discriminator,
                    [x_truth, x_type_data],
                    [x_generated.detach(), particle_type_data.detach()],
                )
                * self.wasserstein_reg
            )

        r1_grad_penalty = 0.0
        if self.r1_reg > 0:
            r1_grad_penalty = (
                get_r1_grad_penalty(
                    self.discriminator,
                    [x_truth, x_type_data.to(x_truth.dtype)],
                )
                * self.r1_reg
            )
        return r1_grad_penalty, wasserstein_grad_penalty

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Common steps for valiation and testing"""

        cond_info, x_momenta, x_type_indices = batch
        num_evts, _ = x_momenta.shape
        # generate events from the Generator
        noise = self.generate_noise(num_evts).to(x_momenta.device)
        particle_kinematics, particle_types = self(noise, cond_info)
        particle_type_idx = torch.argmax(particle_types, dim=1).reshape(num_evts, -1)
        particle_types = particle_types.reshape(num_evts, -1)

        output_particles = []
        if x_type_indices is not None:
            for pidx in range(self.hparams.num_output_particles):
                pidx_start = pidx * self.hparams.num_particle_ids
                pidx_end = pidx_start + self.hparams.num_particle_ids
                gen_types = particle_types[:, pidx_start:pidx_end]

                gen_types = gen_types.detach().cpu().numpy()
                output_particles.append(gen_types.argmax(axis=1))

        fake_output_particles = get_one_hot(
            np.stack(output_particles, axis=1), self.hparams.num_particle_ids
        ).reshape(output_particles[0].shape[0], -1)
        true_output_particles = get_one_hot(
            x_type_indices.detach().cpu().numpy(), self.hparams.num_particle_ids
        ).reshape(x_type_indices.shape[0], -1)

        particle_swd = ot.sliced_wasserstein_distance(
            fake_output_particles, true_output_particles, n_projections=1000
        )

        predictions = (
            torch.cat([particle_kinematics, particle_type_idx], dim=1)
            .cpu()
            .detach()
            .numpy()
        )
        truths = torch.cat([x_momenta, x_type_indices], dim=1).cpu().detach().numpy()

        kinematic_swd = ot.sliced_wasserstein_distance(
            particle_kinematics.detach().cpu().numpy(),
            x_momenta.detach().cpu().numpy(),
            n_projections=100,
        )

        swd = ot.sliced_wasserstein_distance(
            np.concatenate(
                [particle_kinematics.detach().cpu().numpy(), fake_output_particles],
                axis=1,
            ),
            np.concatenate(
                [x_momenta.detach().cpu().numpy(), true_output_particles], axis=1
            ),
            n_projections=100,
        )

        return {
            "swd": swd,
            "particle_swd": particle_swd,
            "kinematic_swd": kinematic_swd,
            "preds": predictions,
            "truths": truths,
        }

    def compare(self, predictions, truths, outname) -> None:
        """Compare the generated events with the real ones
        Parameters:
            perf: dictionary from the step function
        """
        if self.comparison_fn is not None:
            # compare the generated events with the real ones
            images = self.comparison_fn(predictions, truths, outname)
            if self.logger and self.logger.experiment is not None:
                log_images(
                    self.logger,
                    "Particle GAN",
                    images=list(images.values()),
                    caption=list(images.keys()),
                )

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step"""
        perf = self.step(batch, batch_idx)
        swd_distance = perf["swd"]
        particle_swd = perf["particle_swd"]
        kinematic_swd = perf["kinematic_swd"]

        # update and log metrics
        self.val_swd(swd_distance)
        self.val_kinematic_swd(kinematic_swd)
        self.val_particle_swd(particle_swd)

        self.val_min_avg_swd(swd_distance)
        self.val_min_avg_particle_swd(particle_swd)
        self.val_min_avg_kinematic_swd(kinematic_swd)
        self.log("val/swd", swd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/type_swd", particle_swd, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/kinematic_swd",
            kinematic_swd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log("val/min_avg_swd", self.val_min_avg_swd.compute(), prog_bar=True)
        self.log(
            "val/min_avg_kin_swd",
            self.val_min_avg_kinematic_swd.compute(),
            prog_bar=True,
        )
        self.log(
            "val/min_avg_type_swd",
            self.val_min_avg_particle_swd.compute(),
            prog_bar=True,
        )

        if (
            not self.hparams.save_only_improved_plots
            or swd_distance <= self.val_min_avg_swd.compute()
            or kinematic_swd <= self.val_kinematic_swd.compute()
            or particle_swd <= self.val_particle_swd.compute()
        ):
            outname = f"val-{self.current_epoch:02d}-{batch_idx:02d}"
            predictions = perf["preds"]
            truths = perf["truths"]
            self.compare(predictions, truths, outname)

        return perf, batch_idx

    def validaton_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `validation_step()`
        # Need it in multiple GPUs training.
        pass

    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        perf = self.step(batch, batch_idx)
        swd_distance = perf["swd"]
        particle_swd = perf["particle_swd"]
        kinematic_swd = perf["kinematic_swd"]

        # update and log metrics
        self.test_swd(swd_distance)
        self.test_particle_swd(particle_swd)
        self.test_kinematic_swd(kinematic_swd)

        self.test_swd_best(swd_distance)
        self.test_particle_swd_best(particle_swd)
        self.test_kinematic_swd_best(kinematic_swd)
        self.log("test/swd", swd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/particle_swd",
            particle_swd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/kinematic_swd",
            kinematic_swd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/swd_best", self.test_swd_best.compute(), prog_bar=True)
        self.log("test/wd_best", self.test_kinematic_swd_best.compute(), prog_bar=True)
        self.log("test/wd_best", self.test_particle_swd_best.compute(), prog_bar=True)
        # comparison
        if (
            not self.hparams.save_only_improved_plots
            or swd_distance <= self.test_swd_best.compute()
            or kinematic_swd <= self.test_kinematic_swd_best.compute()
            or particle_swd <= self.test_particle_swd_best.compute()
        ):
            outname = f"test-{self.current_epoch:02d}-{batch_idx:02d}"
            predictions = perf["preds"]
            truths = perf["truths"]
            self.compare(predictions, truths, outname)

        return perf, batch_idx
