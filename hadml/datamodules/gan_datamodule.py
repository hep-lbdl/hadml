from typing import Any, Dict, Optional, Tuple, Protocol

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data.dataset import Dataset as GeometricDataset

class GANDataProtocol(Protocol):
    """Define a protocol for GAN data modules."""

    def prepare_data(self) -> None:
        """Prepare data for training and validation.
        Before the create_dataset function is called."""

    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dataset from core dataset.
        Returns:
            torch.Tensor: conditioinal information
            torch.Tensor: particle kinematics
            torch.Tensor: particle types
        """

class ParticleGANDataModule(LightningDataModule):
    def __init__(
        self,
        core_dataset: GANDataProtocol,
        batch_size: int = 5000,
        num_workers: int = 12,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['core_dataset'])
        self.core_dataset = core_dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # read the original file, determine the number of particle types
        # and create a map.
        self.core_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TensorDataset(*self.core_dataset.create_dataset())

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.core_dataset.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

class EventGANDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: GeometricDataset,
        batch_size: int = 500,
        train_val_test_split: Tuple[int, int, int] = (5_000, 1_000, 1_000),
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['core_dataset'])
        self.dataset = dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:

            self.data_train, self.data_val, self.data_test, _ = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split + [len(self.dataset)-sum(self.hparams.train_val_test_split)],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return GeometricDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
