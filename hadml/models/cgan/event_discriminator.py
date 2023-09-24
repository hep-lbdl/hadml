from typing import Any, List, Optional, Dict, Callable, Tuple
import os

import torch
from pytorch_lightning import LightningModule
from scipy import stats
from torchmetrics import MinMetric, MeanMetric
import numpy as np

from hadml.metrics.media_logger import log_images
from hadml.models.components.transform import InvsBoost
from hadml.utils.utils import conditional_cat


class EventDiscModule(LightningModule):
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
        scheduler_generator: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_discriminator: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        comparison_fn: Optional[Callable] = None,
        outdir: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "generator",
                "discriminator",
                "generator_prescale",
                "generator_postscale",
                "discriminator_prescale",
                "comparison_fn",
                "criterion",
            ],
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
        self.val_loss_disc = MeanMetric()
        self.test_loss_disc = MeanMetric()
        self.val_wd = MeanMetric()
        self.val_nll = MeanMetric()

        # for tracking best so far
        self.val_min_avg_wd = MinMetric()
        self.val_min_avg_nll = MinMetric()

        self.test_wd = MeanMetric()
        self.test_nll = MeanMetric()

    def forward(
        self, cond_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            len(cond_info), self.hparams.noise_dim, device=cond_info.device
        )
        cond_info = self.generator_prescale(cond_info)
        x_fake = conditional_cat(cond_info, noise, dim=1)
        fakes = self.generator(x_fake)
        fakes = self.generator_postscale(fakes)
        return fakes

    def configure_optimizers(self):
        opt_gen = self.hparams.optimizer_generator(
            params=self.generator.parameters()
        )  # type: ignore
        opt_disc = self.hparams.optimizer_discriminator(
            params=self.discriminator.parameters()
        )  # type: ignore

        # define schedulers
        if self.hparams.scheduler_generator is not None:
            sched_gen = self.hparams.scheduler_generator(optimizer=opt_gen)
            sched_disc = self.hparams.scheduler_discriminator(optimizer=opt_disc)

            return (
                {
                    "optimizer": opt_disc,
                    "lr_scheduler": {
                        "scheduler": sched_disc,
                        "monitor": "val/min_avg_wd",
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
                        "monitor": "val/min_avg_wd",
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

    def on_train_start(self):
        # By default lightning runs validation sanity checks before training
        # So we need to ensure val_acc_best excludes sanity check accuracy
        self.val_wd.reset()
        self.val_nll.reset()
        self.val_min_avg_wd.reset()
        self.val_min_avg_nll.reset()

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        x_generated = batch["cond_data"].hadrons.reshape((-1, 4))
        x_truth = batch["obs_data"].hadrons.reshape((-1, 4))
        generated_event_label = batch["cond_data"].batch.repeat_interleave(
            batch["cond_data"].hadrons.shape[1] // 4
        )
        observed_event_label = batch["obs_data"].batch.repeat_interleave(
            batch["obs_data"].hadrons.shape[1] // 4
        )
        ptypes_generated = torch.flatten(batch["cond_data"].ptypes, start_dim=0, end_dim=1)
        ptypes_truths = torch.flatten(batch["obs_data"].ptypes, start_dim=0, end_dim=1)
        if len(ptypes_generated.shape)==1:
            ptypes_generated = ptypes_generated.unsqueeze(1)
            ptypes_truths = ptypes_truths.unsqueeze(1)
        

        # generate fake batch
        # angles_generated = self(cluster)
        # x_generated = InvsBoost(cluster, angles_generated).reshape((-1, 4))

        if optimizer_idx == 0:
            return self._discriminator_step(
                x_truth, x_generated, observed_event_label, generated_event_label, ptypes_truths, ptypes_generated
            )

        # if optimizer_idx == 1:
        #     return self._generator_step(x_generated, generated_event_label)

    def _generator_step(self, x_generated, generated_event_label):
        x_generated = self.discriminator_prescale(x_generated)
        score_fakes = self.discriminator(x_generated, generated_event_label).squeeze(-1)
        label = torch.ones_like(score_fakes)
        loss_gen = self.criterion(score_fakes, label)

        # update and log metrics
        self.train_loss_gen(loss_gen)
        self.log("lossG", loss_gen, prog_bar=True)
        return {"loss": loss_gen}

    def _discriminator_step(
        self, x_truth, x_generated, observed_event_label, generated_event_label, ptypes_truths, ptypes_generated
    ):
        # create input
        x_truth = self.discriminator_prescale(x_truth)
        x_generated = self.discriminator_prescale(x_generated)
        input_generated = torch.cat([x_generated, ptypes_generated], dim=-1)
        input_truths = torch.cat([x_truth, ptypes_truths], dim=-1)

        # with real batch
        score_truth = self.discriminator(input_truths, observed_event_label).squeeze(-1)
        label = torch.ones_like(score_truth)
        loss_real = self.criterion(score_truth, label)

        # with fake batch
        score_fakes = self.discriminator(input_generated, generated_event_label).squeeze(-1)
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
        # cluster = batch["cond_data"].cluster
        # angles_truths = batch["obs_data"].x
        hadrons_generated = batch["cond_data"].hadrons.reshape((-1, 4))
        hadrons_truths = batch["obs_data"].hadrons.reshape((-1, 4))
        generated_event_label = batch["cond_data"].batch.repeat_interleave(
            batch["cond_data"].hadrons.shape[1] // 4
        )
        observed_event_label = batch["obs_data"].batch.repeat_interleave(
            batch["obs_data"].hadrons.shape[1] // 4
        )

        # Append particle pids
        ptypes_generated = torch.flatten(batch["cond_data"].ptypes, start_dim=0, end_dim=1)
        ptypes_truths = torch.flatten(batch["obs_data"].ptypes, start_dim=0, end_dim=1)
        if len(ptypes_generated.shape)==1:
            ptypes_generated = ptypes_generated.unsqueeze(1)
            ptypes_truths = ptypes_truths.unsqueeze(1)
        hadrons_truths = self.discriminator_prescale(hadrons_truths)
        hadrons_generated = self.discriminator_prescale(hadrons_generated)
        input_generated = torch.cat([hadrons_generated, ptypes_generated], dim=-1)
        input_truths = torch.cat([hadrons_truths, ptypes_truths], dim=-1)

        # with real batch
        score_truth = self.discriminator(input_truths, observed_event_label).squeeze(-1)
        label = torch.ones_like(score_truth)
        loss_real = self.criterion(score_truth, label)

        # with fake batch
        score_fakes = self.discriminator(input_generated, generated_event_label).squeeze(-1)
        fake_labels = torch.zeros_like(score_fakes)
        loss_fake = self.criterion(score_fakes, fake_labels)

        loss_disc = (loss_real + loss_fake) / 2

        # compute the WD for the particle kinmatics
        score_fakes = score_fakes.cpu().detach().numpy()
        score_truth = score_truth.cpu().detach().numpy()
        hadrons_generated = hadrons_generated.cpu().detach().numpy()
        hadrons_truths = hadrons_truths.cpu().detach().numpy()
        generated_event_label = generated_event_label.cpu().detach().numpy()
        observed_event_label = observed_event_label.cpu().detach().numpy()
        ptypes_generated = ptypes_generated.cpu().detach().numpy()
        ptypes_truths = ptypes_truths.cpu().detach().numpy()

        return {
            "ls": loss_disc,
            "hadrons_generated": hadrons_generated,
            "hadrons_truths": hadrons_truths,
            "generated_event_label": generated_event_label,
            "observed_event_label": observed_event_label,
            "score_generated": score_fakes,
            "score_truths": score_truth,
            "ptypes_generated": ptypes_generated,
            "ptypes_truths": ptypes_truths
        }

    def compare(
        self,
        score_generated,
        score_truths,
        outname,
    ) -> None:
        """Compare the generated events with the real ones
        Parameters:
            perf: dictionary from the step function
        """
        if self.comparison_fn is not None:
            # compare the generated events with the real ones
            images = self.comparison_fn(
                score_generated,
                score_truths,
                outname,
            )
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
        # wd_distance = perf["wd"]
        # avg_nll = perf["nll"]
        # self.val_wd(wd_distance)
        # self.val_nll(avg_nll)

        return perf

    def validation_epoch_end(self, validation_step_outputs):
        outname = f"val-{self.current_epoch}"
        hadrons_generated = []
        hadrons_truths = []
        score_generated = []
        score_truths = []
        generated_event_label = []
        observed_event_label = []
        ptypes_generated = []
        ptypes_truths = []
        for perf in validation_step_outputs:
            self.val_loss_disc(perf["ls"])
            hadrons_generated = (
                perf["hadrons_generated"]
                if len(hadrons_generated) == 0
                else np.concatenate((hadrons_generated, perf["hadrons_generated"]))
            )
            hadrons_truths = (
                perf["hadrons_truths"]
                if len(hadrons_truths) == 0
                else np.concatenate((hadrons_truths, perf["hadrons_truths"]))
            )
            score_generated = (
                perf["score_generated"]
                if len(score_generated) == 0
                else np.concatenate((score_generated, perf["score_generated"]))
            )
            score_truths = (
                perf["score_truths"]
                if len(score_truths) == 0
                else np.concatenate((score_truths, perf["score_truths"]))
            )
            generated_event_label = (
                perf["generated_event_label"]
                if len(generated_event_label) == 0
                else np.concatenate(
                    (
                        generated_event_label,
                        perf["generated_event_label"]
                        + generated_event_label[-1]
                        + 1,
                    )
                )
            )
            observed_event_label = (
                perf["observed_event_label"]
                if len(observed_event_label) == 0
                else np.concatenate(
                    (
                        observed_event_label,
                        perf["observed_event_label"]
                        + observed_event_label[-1]
                        + 1
                    )
                )
            )

            ptypes_generated = (
                perf["ptypes_generated"]
                if len(ptypes_generated) == 0
                else np.concatenate((ptypes_generated, perf["ptypes_generated"]))
            )
            ptypes_truths = (
                perf["ptypes_truths"]
                if len(ptypes_truths) == 0
                else np.concatenate((ptypes_truths, perf["ptypes_truths"]))
            )
        
        self.log("val/loss", self.val_loss_disc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_disc.reset()
        self.compare(
            score_generated,
            score_truths,
            outname,
        )

        if self.current_epoch == 0:
            os.makedirs(self.hparams.outdir, exist_ok=True)
            np.savez_compressed(os.path.join(self.hparams.outdir, "initial.npz"),
                                score_generated=score_generated,
                                score_truths=score_truths,
                                hadrons_generated=hadrons_generated,
                                hadrons_truths=hadrons_truths,
                                generated_event_label=generated_event_label,
                                observed_event_label=observed_event_label,
                                ptypes_generated=ptypes_generated,
                                ptypes_truths=ptypes_truths)
        if self.current_epoch == self.trainer.max_epochs - 1:
            os.makedirs(self.hparams.outdir, exist_ok=True)
            np.savez_compressed(os.path.join(self.hparams.outdir, "final.npz"),
                                score_generated=score_generated,
                                score_truths=score_truths,
                                hadrons_generated=hadrons_generated,
                                hadrons_truths=hadrons_truths,
                                generated_event_label=generated_event_label,
                                observed_event_label=observed_event_label,
                                ptypes_generated=ptypes_generated,
                                ptypes_truths=ptypes_truths)

    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        perf = self.step(batch, batch_idx)
        # wd_distance = perf["wd"]
        # avg_nll = perf["nll"]
        # self.test_wd(wd_distance)
        # self.test_nll(avg_nll)

        return perf

    def test_epoch_end(self, test_step_outputs):
        outname = f"test-{self.current_epoch}"
        hadrons_generated = []
        hadrons_truths = []
        score_generated = []
        score_truths = []
        generated_event_label = []
        observed_event_label = []
        ptypes_generated = []
        ptypes_truths = []
        for perf in test_step_outputs:
            self.test_loss_disc(perf["ls"])
            hadrons_generated = (
                perf["hadrons_generated"]
                if len(hadrons_generated) == 0
                else np.concatenate((hadrons_generated, perf["hadrons_generated"]))
            )
            hadrons_truths = (
                perf["hadrons_truths"]
                if len(hadrons_truths) == 0
                else np.concatenate((hadrons_truths, perf["hadrons_truths"]))
            )
            score_generated = (
                perf["score_generated"]
                if len(score_generated) == 0
                else np.concatenate((score_generated, perf["score_generated"]))
            )
            score_truths = (
                perf["score_truths"]
                if len(score_truths) == 0
                else np.concatenate((score_truths, perf["score_truths"]))
            )
            generated_event_label = (
                perf["generated_event_label"]
                if len(generated_event_label) == 0
                else np.concatenate(
                    (
                        generated_event_label,
                        perf["generated_event_label"]
                        + generated_event_label[-1]
                        + 1,
                    )
                )
            )
            observed_event_label = (
                perf["observed_event_label"]
                if len(observed_event_label) == 0
                else np.concatenate(
                    (
                        observed_event_label,
                        perf["observed_event_label"]
                        + observed_event_label[-1]
                        + 1
                    )
                )
            )
            ptypes_generated = (
                perf["ptypes_generated"]
                if len(ptypes_generated) == 0
                else np.concatenate((ptypes_generated, perf["ptypes_generated"]))
            )
            ptypes_truths = (
                perf["ptypes_truths"]
                if len(ptypes_truths) == 0
                else np.concatenate((ptypes_truths, perf["ptypes_truths"]))
            )
        
        self.log("test/loss", self.test_loss_disc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.test_loss_disc.reset()
        self.compare(
            score_generated,
            score_truths,
            outname,
        )

        os.makedirs(self.hparams.outdir, exist_ok=True)
        np.savez_compressed(os.path.join(self.hparams.outdir, "best.npz"),
                            score_generated=score_generated,
                            score_truths=score_truths,
                            hadrons_generated=hadrons_generated,
                            hadrons_truths=hadrons_truths,
                            generated_event_label=generated_event_label,
                            observed_event_label=observed_event_label,
                            ptypes_generated=ptypes_generated,
                            ptypes_truths=ptypes_truths)
        
