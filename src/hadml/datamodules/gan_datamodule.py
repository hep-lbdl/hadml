from typing import Any, Dict, Optional, Protocol, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch_geometric.data.dataset import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

if pl.__version__ <= "2.0.0":
    from pytorch_lightning.trainer.supporters import CombinedLoader
else:
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

from hadml.datamodules.components.utils import get_num_asked_events, process_data_split


class GANDataProtocol(Protocol):
    """Define a protocol for GAN data modules."""

    def prepare_data(self) -> None:
        """Prepare data for training and validation.
        Before the create_dataset function is called.
        """

    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dataset from core dataset.

        Returns
        -------
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
        drop_last: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["core_dataset"])
        self.core_dataset = core_dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # read the original file, determine the number of particle types
        # and create a map.
        self.core_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables:
            `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning
        with both `trainer.fit()` and `trainer.test()`
        so be careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TensorDataset(*self.core_dataset.create_dataset())

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.core_dataset.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Number of training examples: {len(self.data_train)}")
            print(f"Number of validation examples: {len(self.data_val)}")
            print(f"Number of test examples: {len(self.data_test)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""


class EventGANDataModule(LightningDataModule):
    def __init__(
        self,
        cond_dataset: GeometricDataset,
        obs_dataset: GeometricDataset,
        batch_size: int = 500,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        frac_data_used: Optional[float] = 1.0,
        examples_used: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["cond_dataset", "obs_dataset"])

        self.hparams.examples_used = process_data_split(
            examples_used, frac_data_used, train_val_test_split
        )

        num_asked_events_obs = get_num_asked_events(
            self.hparams.examples_used, frac_data_used, len(obs_dataset)
        )
        num_asked_events_cond = get_num_asked_events(
            self.hparams.examples_used, frac_data_used, len(cond_dataset)
        )
        num_asked_events = min(num_asked_events_obs, num_asked_events_cond)
        print(f"Asking for smaller number of events ({num_asked_events})...")
        self.cond_dataset = cond_dataset[:num_asked_events]
        self.obs_dataset = obs_dataset[:num_asked_events]

        self.cond_data_train: Optional[Dataset] = None
        self.cond_data_val: Optional[Dataset] = None
        self.cond_data_test: Optional[Dataset] = None

        self.obs_data_train: Optional[Dataset] = None
        self.obs_data_val: Optional[Dataset] = None
        self.obs_data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables:
            `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning
        with both `trainer.fit()` and `trainer.test()`
        so be careful not to execute things like random split twice!
        """
        if not self.cond_data_train and not self.cond_data_val and not self.cond_data_test:
            (
                self.cond_data_train,
                self.cond_data_val,
                self.cond_data_test,
            ) = random_split(
                dataset=self.cond_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        if not self.obs_data_train and not self.obs_data_val and not self.obs_data_test:
            (
                self.obs_data_train,
                self.obs_data_val,
                self.obs_data_test,
            ) = random_split(
                dataset=self.obs_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Number of training examples: {len(self.obs_data_train)}")
            print(f"Number of validation examples: {len(self.obs_data_val)}")
            print(f"Number of test examples: {len(self.obs_data_test)}")

    def train_dataloader(self):
        return {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            ),
        }

    def val_dataloader(self):
        loaders = {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        loaders = {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")
