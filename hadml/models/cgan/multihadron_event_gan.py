import torch
from pytorch_lightning import LightningModule
from utils.utils import conditional_cat
from typing import Optional, Callable
from metrics.media_logger import log_images
from torch.optim import Optimizer


class MultiHadronEventGANModule(LightningModule):
    def __init__(
        self,
        datamodule: torch.nn.Module,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: Optimizer,
        optimizer_discriminator: Optimizer,
        noise_dim: int,
        comparison_fn: Optional[Callable]
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, clusters):
        generator_input = conditional_cat(clusters, self._generate_noise(len(clusters)))
        generated_hadrons = self.hparams.generator(generator_input)
        return generated_hadrons
    
    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Debugging: batch_size=1, num_workers=0:
        # batch[0] - generator input
        # batch[1] - ideal generator output (discriminator input)
        return None

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        generator_opt = self.hparams.optimizer_generator(params=self.generator.parameters())
        disriminator_opt = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())
        return generator_opt, disriminator_opt
    
    def _generate_noise(self, n_tokens):
        return torch.randn(n_tokens, self.hparams.noise_dim)
    
    def _compare(self, predictions, truths):
        images = self.hparams.comparison_fn(predictions, truths)
        # Attributes self.logger and self.logger.experiment are defined by the logger passed
        # to the trainer which in turn uses an object of this model class: 
        if self.logger and self.logger.experiment is not None:
            log_images(logger=self.logger, key="MultiHadronEvent GAN",
                       images=list(images.values()), caption=list(images.keys()))
