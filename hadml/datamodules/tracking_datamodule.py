import os
import numpy as np

from typing import Tuple, Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from hadml.datamodules.components.utils import process_data_split, get_num_asked_events


class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        input_dir,
        train_val_test_split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        frac_data_used: Optional[float] = 1.0,
        examples_used: Optional[int] = None,
        num_workers: int = 12,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.hparams.examples_used = process_data_split(
            examples_used, frac_data_used, train_val_test_split
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        input_dir = self.hparams.input_dir
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])

        num_tot_events = len(all_events)
        num_asked_events = get_num_asked_events(
            self.hparams.examples_used, self.hparams.frac_data_used, num_tot_events
        )

        def read_fn(path):
            from torch_geometric.data import Data

            store = np.load(path)
            #     x=torch.from_numpy(np.concatenate([store['x'], store['cells']], axis=1)).float(),
            data = Data(
                x=torch.from_numpy(store["x"]).float(),
                true_edges=torch.from_numpy(store["true_edges"]),
            )
            return data

        loaded_events = [read_fn(event) for event in all_events[:num_asked_events]]
        self.data_train, self.data_val, self.data_test = random_split(
            loaded_events,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=1,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
