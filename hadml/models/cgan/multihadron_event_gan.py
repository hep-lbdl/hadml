import torch
from pytorch_lightning import LightningModule
from utils.utils import conditional_cat


class MultiHadronEventGANModule(LightningModule):
    def __init__(
        self,
        datamodule: torch.nn.Module,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        noise_dim: int
    ):
        super().__init__()
        self.datamodule = datamodule
        self.generator = generator      
        self.discriminator = discriminator
        self.noise_dim = noise_dim

    def forward(self, clusters):
        generator_input = conditional_cat(clusters, self._generate_noise(len(clusters)))
        generated_hadrons = self.generator(generator_input)
        return generated_hadrons
    
    def setup():
        pass

    def training_step(self, batch, batch_idx):
        return batch.mean()

    def validation_step():
        pass

    def test_step():
        pass

    def predict_step():
        pass

    def configure_optimizers():
        return torch.optim.Adam()
    
    def _generate_noise(self, n_tokens):
        return torch.randn(n_tokens, self.noise_dim)