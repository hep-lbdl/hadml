import os
import numpy as np

from typing import Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        input_dir,
        train_val_test_split: Tuple[int, int, int] = (10, 5, 5),
        num_workers: int = 12,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def setup(self):
        input_dir = self.hparams.input_dir
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])

        num_asked_evts = sum(self.hparams.train_val_test_split)
        num_tot_evts = len(all_events)

        print("Use {} events out of total {} events".format(num_asked_evts, num_tot_evts))
        if num_asked_evts > len(all_events):
            raise ValueError(f"Number of events {num_tot_evts} is less than asked {num_asked_evts}")

        def read_fn(path):
            from torch_geometric.data import Data
            store = np.load(path)
        #     x=torch.from_numpy(np.concatenate([store['x'], store['cells']], axis=1)).float(),
            data = Data(
                x=torch.from_numpy(store['x']).float(),
                true_edges=torch.from_numpy(store['true_edges']),
            )
            return data

        loaded_events = [read_fn(event) for event in all_events[:num_asked_evts]]
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
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )
