from typing import Any, List, Optional, Dict, Callable, Tuple

import torch
from pytorch_lightning import LightningModule
from scipy import stats
from torchmetrics import MinMetric, MeanMetric

from hadml.metrics.media_logger import log_images
from hadml.models.components.transform import InvsBoost
from hadml.utils.utils import conditional_cat


class CondEventGANModule(LightningModule):
    """Event GAN module to generate events.
    The conditional inputs feeding to the gnerator are cluster's 4 vector.
    The generator will generate kinematics of the outgoing particles.

    The discriminator will take the generated events and the real events
    as inputs, and output a probability of the generated events being real.

    Have not considered the particle types for now.

    Parameters:
        noise_dim: dimension of noise vector
        generator: generator network
        discriminator: discriminator network
        optimizer_generator: generator optimizer
        optimizer_discriminator: discriminator optimizer
        comparison_fn: function to compare generated and real data
    """

    def __init__(
        self,
        noise_dim: int,
        cond_info_dim: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        generator_prescale: torch.nn.Module,
        generator_postscale: torch.nn.Module,
        discriminator_prescale: torch.nn.Module,
        num_critics: int,
        num_gen: int,
        criterion: torch.nn.Module,
        scheduler_generator: Optional[
            torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_discriminator: Optional[
            torch.optim.lr_scheduler._LRScheduler] = None,
        comparison_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["generator", "discriminator", "generator_prescale",
                    "generator_postscale", "discriminator_prescale",
                    "comparison_fn", "criterion"],
        )

        self.generator = generator
        self.discriminator = discriminator
        self.generator_prescale = generator_prescale
        self.generator_postscale = generator_postscale
        self.discriminator_prescale = discriminator_prescale
        self.comparison_fn = comparison_fn

        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss_gen = MeanMetric()
        self.train_loss_disc = MeanMetric()
        self.val_wd = MeanMetric()
        self.val_nll = MeanMetric()

        # for tracking best so far
        self.val_min_avg_wd = MinMetric()
        self.val_min_avg_nll = MinMetric()

        self.test_wd = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_wd_best = MinMetric()
        self.test_nll_best = MinMetric()

    def forward(
        self, cond_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            len(cond_info), self.hparams.noise_dim, device=cond_info.device)
        cond_info = self.generator_prescale(cond_info)
        x_fake = conditional_cat(cond_info, noise, dim=1)
        fakes = self.generator(x_fake)
        fakes = self.generator_postscale(fakes)
        return fakes

    def configure_optimizers(self):
        opt_gen = self.hparams.optimizer_generator(
            params=self.generator.parameters())  # type: ignore
        opt_disc = self.hparams.optimizer_discriminator(
            params=self.discriminator.parameters())  # type: ignore

        # define schedulers
        if self.hparams.scheduler_generator is not None:
            sched_gen = self.hparams.scheduler_generator(optimizer=opt_gen)
            sched_disc = self.hparams.scheduler_discriminator(
                optimizer=opt_disc)

            return (
                {
                    "optimizer": opt_disc,
                    "lr_scheduler": {
                        "scheduler": sched_disc,
                        "monitor": "val/min_avg_wd",
                        "interval": "step",
                        "frequency": self.trainer.val_check_interval,
                        "strict": True,
                        "name": "DiscriminatorLRScheduler"
                    },
                    "frequency": self.hparams.num_critics
                },
                {
                    "optimizer": opt_gen,
                    "lr_scheduler": {
                        "scheduler": sched_gen,
                        "monitor": "val/min_avg_wd",
                        "interval": "step",
                        "frequency": self.trainer.val_check_interval,
                        "strict": True,
                        "name": "GeneratorLRScheduler"
                    },
                    "frequency": self.hparams.num_gen
                })

        return (
            {
                "optimizer": opt_disc,
                "frequency": self.hparams.num_critics
            }, {
                "optimizer": opt_gen,
                "frequency": self.hparams.num_gen
            })

    def on_train_start(self):
        # By default lightning runs validation sanity checks before training
        # So we need to ensure val_acc_best excludes sanity check accuracy
        self.val_min_avg_wd.reset()
        self.val_min_avg_nll.reset()
        self.test_wd_best.reset()
        self.test_nll_best.reset()

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        cluster = batch["cond_data"].cluster
        x_truth = batch["obs_data"].hadrons.reshape((-1, 4))
        generated_event_label = torch.cat((
            batch["cond_data"].batch.reshape(-1, 1),
            batch["cond_data"].batch.reshape(-1, 1)), dim=1).reshape(-1)
        observed_event_label = torch.cat((
            batch["obs_data"].batch.reshape(-1, 1),
            batch["obs_data"].batch.reshape(-1, 1)), dim=1).reshape(-1)

        # generate fake batch
        angles_generated = self(cluster)
        x_generated = InvsBoost(cluster, angles_generated).reshape((-1, 4))

        if optimizer_idx == 0:
            return self._discriminator_step(
                x_truth, x_generated, observed_event_label,
                generated_event_label)

        if optimizer_idx == 1:
            return self._generator_step(x_generated, generated_event_label)

    def _generator_step(self, x_generated, generated_event_label):
        x_generated = self.discriminator_prescale(x_generated)
        score_fakes = self.discriminator(
            x_generated, generated_event_label).squeeze(-1)
        label = torch.ones_like(score_fakes)
        loss_gen = self.criterion(score_fakes, label)

        # update and log metrics
        self.train_loss_gen(loss_gen)
        self.log("lossG", loss_gen, prog_bar=True)
        return {"loss": loss_gen}

    def _discriminator_step(self, x_truth, x_generated,
                            observed_event_label, generated_event_label):
        # with real batch
        x_truth = self.discriminator_prescale(x_truth)
        score_truth = self.discriminator(
            x_truth, observed_event_label).squeeze(-1)
        label = torch.ones_like(score_truth)
        loss_real = self.criterion(score_truth, label)

        # with fake batch
        x_generated = self.discriminator_prescale(x_generated)
        score_fakes = self.discriminator(
            x_generated, generated_event_label).squeeze(-1)
        fake_labels = torch.zeros_like(score_fakes)
        loss_fake = self.criterion(score_fakes, fake_labels)

        loss_disc = (loss_real + loss_fake) / 2

        # update and log metrics
        self.train_loss_disc(loss_disc)
        self.log("lossD", loss_disc, prog_bar=True)
        return {"loss": loss_disc}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Common steps for valiation and testing"""
        cluster = batch["cond_data"].cluster
        angles_truths = batch["obs_data"].x
        hadrons_truth = batch["obs_data"].hadrons.reshape((-1, 4))

        # generate events from the Generator
        angles_generated = self(cluster)
        hadrons_generated = InvsBoost(
            cluster, angles_generated).reshape((-1, 4))

        # compute the WD for the particle kinmatics
        angles_predictions = angles_generated.cpu().detach().numpy()
        angles_truths = angles_truths.cpu().detach().numpy()
        hadrons_predictions = hadrons_generated.cpu().detach().numpy()
        hadrons_truth = hadrons_truth.cpu().detach().numpy()

        distances = [
            stats.wasserstein_distance(
                hadrons_predictions[:, idx],
                hadrons_truth[:, idx]) for idx in range(4)
        ]
        wd_distance = sum(distances) / len(distances)

        return {"wd": wd_distance, "nll": 0.,
                "angles_preds": angles_predictions,
                "angles_truths": angles_truths,
                "hadrons_preds": hadrons_predictions,
                "hadrons_truth": hadrons_truth}

    def compare(self, angles_predictions, angles_truths,
                hadrons_predictions, hadrons_truth, outname) -> None:
        """Compare the generated events with the real ones
        Parameters:
            perf: dictionary from the step function
        """
        if self.comparison_fn is not None:
            # compare the generated events with the real ones
            images = self.comparison_fn(
                angles_predictions, angles_truths,
                hadrons_predictions, hadrons_truth, outname)
            if self.logger and self.logger.experiment is not None:
                log_images(
                    self.logger,
                    "Event GAN",
                    images=list(images.values()),
                    caption=list(images.keys()),
                )

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step"""
        perf = self.step(batch, batch_idx)
        wd_distance = perf["wd"]
        avg_nll = perf["nll"]

        # update and log metrics
        self.val_wd(wd_distance)
        self.val_nll(avg_nll)

        self.val_min_avg_wd(wd_distance)
        self.val_min_avg_nll(avg_nll)
        self.log("val/wd", wd_distance, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/nll", avg_nll, on_step=False,
                 on_epoch=True, prog_bar=True)

        self.log("val/min_avg_wd", self.val_min_avg_wd.compute(),
                 prog_bar=True)
        self.log("val/min_avg_nll", self.val_min_avg_nll.compute(),
                 prog_bar=True)

        if (
            avg_nll <= self.val_min_avg_nll.compute()
            or wd_distance <= self.val_min_avg_wd.compute()
        ):
            outname = f"val-{self.current_epoch}-{batch_idx}"
            angles_predictions = perf['angles_preds']
            angles_truths = perf['angles_truths']
            hadrons_predictions = perf['hadrons_preds']
            hadrons_truth = perf['hadrons_truth']
            self.compare(angles_predictions, angles_truths,
                         hadrons_predictions, hadrons_truth, outname)

        return perf, batch_idx

    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        perf = self.step(batch, batch_idx)
        wd_distance = perf["wd"]
        avg_nll = perf["nll"]

        # update and log metrics
        self.test_wd(wd_distance)
        self.test_nll(avg_nll)
        self.test_wd_best(wd_distance)
        self.test_nll_best(avg_nll)

        self.log("test/wd", wd_distance, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/nll", avg_nll, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/wd_best", self.test_wd_best.compute(), prog_bar=True)
        self.log("test/nll_best", self.test_nll_best.compute(), prog_bar=True)
        # comparison
        if (
            avg_nll <= self.test_nll_best.compute()
            or wd_distance <= self.test_wd_best.compute()
        ):
            outname = f"test-{self.current_epoch}-{batch_idx}"
            angles_predictions = perf['angles_preds']
            angles_truths = perf['angles_truths']
            hadrons_predictions = perf['hadrons_preds']
            hadrons_truth = perf['hadrons_truth']
            self.compare(angles_predictions, angles_truths,
                         hadrons_predictions, hadrons_truth, outname)

        return perf, batch_idx
