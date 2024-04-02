from typing import Callable, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric


class ParticleEmbeddingModule(LightningModule):
    """Metric Learning. Embedding nodes into a vector space so that similar nodes are close together."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        comparison_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_min_loss = MinMetric()
        self.test_min_loss = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_min_loss.reset()
        self.test_min_loss.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        _, _, x_type_indices = batch
        num_particles = x_type_indices.shape[1]
        type_encoding = [self.net(x_type_indices[:, idx]) for idx in range(num_particles)]

        # true cases
