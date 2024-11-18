import pathlib
from collections import Counter
import os
import pickle
from typing import Dict, Optional, Tuple
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.mixins import HyperparametersMixin

from hadml.datamodules.components.utils import (
    read_dataframe,
    split_to_float,
    InputScaler,
    boost,
    process_data_split,
    get_num_asked_events,
    get_angles,
)

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Dataset

pid_map_fname = "pids_to_ix.pkl"


class Herwig(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        fname: str = "allHadrons_with_quark.dat",
        cache_dir: Optional[str] = None,
        origin_fname: str = "cluster_ML_allHadrons_10M.txt",
        train_val_test_split: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        frac_data_used: Optional[float] = 1.0,
        examples_used: Optional[int] = None,
        num_output_hadrons: int = 2,
        num_particle_kinematics: int = 2,
        # hadron_type_embedding_dim: int = 10,
        num_used_hadron_types: Optional[int] = None,
    ):
        """This is for the GAN datamodule. It reads clusters from a file"""
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.hparams.examples_used = process_data_split(
            examples_used, frac_data_used, train_val_test_split
        )

        self.cond_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

        self.pids_to_ix: Optional[Dict[int, int]] = None

        # particle type map
        self.pids_map_fname = os.path.join(self.hparams.data_dir, pid_map_fname)
        self.num_hadron_types: int = 0

    def prepare_data(self):
        # read the original file, determine the number of particle types
        # and create a map.
        if os.path.exists(self.pids_map_fname):
            print("Loading existing pids map")
            self.pids_to_ix = pickle.load(open(self.pids_map_fname, "rb"))
            if self.hparams.num_used_hadron_types is None:
                self.num_hadron_types = len(list(self.pids_to_ix.keys()))
            else:
                self.num_hadron_types = self.hparams.num_used_hadron_types
            print("END...Loading existing pids map")
        else:
            fname = os.path.join(self.hparams.data_dir, self.hparams.origin_fname)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} not found.")
            df = pd.read_csv(
                fname, usecols=[3, 4], sep=";", header=None, names=None, engine="c"
            )

            def extract_type(df, sep=","):
                out = df
                if type(df.iloc[0]) == str:
                    out = df.str.extract(f"^([^{sep}]+)").astype(np.int16)
                return out

            h1_type, h2_type = [extract_type(df[idx]) for idx in [3, 4]]
            del df
            all_types = np.concatenate([h1_type, h2_type]).squeeze()
            count = Counter(all_types)
            hadron_pids = list(map(lambda x: x[0], count.most_common()))

            self.pids_to_ix = {pids: i for i, pids in enumerate(hadron_pids)}
            if self.hparams.num_used_hadron_types is None:
                self.num_hadron_types = len(hadron_pids)
            else:
                self.num_hadron_types = self.hparams.num_used_hadron_types

            pickle.dump(self.pids_to_ix, open(self.pids_map_fname, "wb"))

    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """It creates the dataset for training a conditional GAN.
        Returns:
            cond_info: conditional information
            x_truth:   target truth information with conditonal information
        """
        fname = pathlib.Path(self.hparams.data_dir).joinpath(self.hparams.fname)
        cluster, event_labels, h1, h2, q1, q2 = self._load_dataset(fname)

        cluster = cluster[:, [1, 2, 3, 4]]

        h1_types = h1[:, [0]]
        h2_types = h2[:, [0]]
        h1 = h1[:, [1, 2, 3, 4]]
        h2 = h2[:, [1, 2, 3, 4]]

        q1_types = q1[:, [0]]
        q2_types = q2[:, [0]]
        q1 = q1[:, [1, 2, 3, 4]]
        q2 = q2[:, [1, 2, 3, 4]]

        org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
        del cluster, q1, q2, h1, h2

        event_labels, q1_types, q2_types, h1_types, h2_types, org_inputs = self._filter_unnamed_types(
            event_labels, q1_types, q2_types, h1_types, h2_types, org_inputs
        )

        num_tot_evts = len(org_inputs)
        num_asked_events = get_num_asked_events(
            self.hparams.examples_used, self.hparams.frac_data_used, num_tot_evts
        )
        org_inputs = org_inputs[:num_asked_events]
        q1_types = q1_types[:num_asked_events]
        q2_types = q2_types[:num_asked_events]
        h1_types = h1_types[:num_asked_events]
        h2_types = h2_types[:num_asked_events]
        event_labels = torch.from_numpy(event_labels)[:num_asked_events]

        new_inputs = boost(org_inputs)

        phi, theta = get_angles(new_inputs[:, -4:])
        theta = theta + math.pi * (theta < 0)

        true_hadron_angles = np.stack([phi, theta], axis=1)
        # hadron_angles_prescaler = MinMaxScaler((-1, 1))
        # true_hadron_angles = hadron_angles_prescaler.fit_transform(true_hadron_angles)
        true_hadron_angles = torch.from_numpy(true_hadron_angles.astype(np.float32))
        true_hadron_momenta = torch.from_numpy(org_inputs[:, -8:])

        q_phi, q_theta = get_angles(new_inputs[:, 4:8])
        q_momenta = np.stack([q_phi, q_theta], axis=1)
        cond_info = np.concatenate([org_inputs[:, :4], q1_types, q2_types, q_momenta], axis=1)
        # cond_info_prescaler = MinMaxScaler((-1, 1))
        # cond_info = cond_info_prescaler.fit_transform(cond_info)
        cond_info = torch.from_numpy(cond_info.astype(np.float32))

        # convert particle IDs to indices
        # then these indices can be embedded in N dim. space
        h1_type_indices = np.vectorize(self.pids_to_ix.get)(h1_types.astype(np.int16))
        h2_type_indices = np.vectorize(self.pids_to_ix.get)(h2_types.astype(np.int16))
        target_hadron_types_idx = torch.from_numpy(
            np.concatenate([h1_type_indices, h2_type_indices], axis=1)
        )

        # joblib.dump(cond_info_prescaler, f'{self.hparams.data_dir}/cond_info_prescaler.gz')
        # joblib.dump(hadron_angles_prescaler, f'{self.hparams.data_dir}/hadron_angles_prescaler.gz')

        dataset = (
            cond_info,
            true_hadron_angles,
            target_hadron_types_idx,
            true_hadron_momenta,
            event_labels,
        )

        if self.hparams.num_used_hadron_types is not None:
            used_idx = target_hadron_types_idx < self.hparams.num_used_hadron_types
            used_idx = used_idx.sum(axis=1).eq(used_idx.shape[1])
            print(
                f"{1 - used_idx.to(torch.float32).mean():.3f} of all training examples were dropped due to not all particle types being used."
            )
            dataset = tuple(table[used_idx] for table in dataset)

        self.summarize()
        return dataset

    def _filter_unnamed_types(self, event_labels, q1_types, q2_types, h1_types, h2_types, org_inputs):
        mask = (np.isin(h1_types.reshape(-1), list(self.pids_to_ix.keys()))) & (
            np.isin(h2_types.reshape(-1), list(self.pids_to_ix.keys()))
        )
        event_labels = event_labels[mask]
        org_inputs = org_inputs[mask]
        q1_types = q1_types[mask]
        q2_types = q2_types[mask]
        h1_types = h1_types[mask]
        h2_types = h2_types[mask]
        return event_labels, q1_types, q2_types, h1_types, h2_types, org_inputs

    def _load_dataset(self, fname):
        load_from_cache = False
        processed_fnames = None
        if self.hparams.cache_dir is not None:
            processed_fnames = {
                name: pathlib.Path(self.hparams.cache_dir).joinpath(
                    f"{name}.npy"
                )
                for name in ["q1", "q2", "c", "h1", "h2", "event_labels"]
            }
            load_from_cache = all(
                [os.path.exists(f) for f in processed_fnames.values()]
            )
        if not load_from_cache:
            cluster, event_labels, h1, h2, q1, q2 = self._load_raw_dataset(fname)

            if processed_fnames is not None:
                self._save_data_arrays(event_labels, h1, h2, processed_fnames, q1, q2, cluster)
        else:
            print(f"Loading cached data arrays...")
            cluster, event_labels, h1, h2, q1, q2 = self._load_data_arrays(
                processed_fnames
            )
        return cluster, event_labels, h1, h2, q1, q2

    def _load_data_arrays(self, processed_fnames):
        q1 = np.load(processed_fnames["q1"])
        q2 = np.load(processed_fnames["q2"])
        cluster = np.load(processed_fnames["c"])
        h1 = np.load(processed_fnames["h1"])
        h2 = np.load(processed_fnames["h2"])
        event_labels = np.load(processed_fnames["event_labels"])
        return cluster, event_labels, h1, h2, q1, q2

    def _save_data_arrays(self, event_labels, h1, h2, processed_fnames, q1, q2, cluster):
        for name, array in zip(
            ["q1", "q2", "c", "h1", "h2", "event_labels"],
            [q1, q2, cluster, h1, h2, event_labels],
        ):
            print(f"Saving {name}...")
            np.save(processed_fnames[name], array)

    def _load_raw_dataset(self, fname):
        clusters = []
        q1, q2, c, h1, h2 = [], [], [], [], []
        event_labels = []
        labels_tmp = []
        with open(fname) as f:
            for i, event_line in enumerate(f):
                if i % 100000 == 0:
                    print(i)
                if i % 5000000 == 0 and i > 0:
                    df = pd.DataFrame(clusters)
                    q1.append(split_to_float(df[0]))
                    q2.append(split_to_float(df[1]))
                    c.append(split_to_float(df[2]))
                    h1.append(split_to_float(df[3]))
                    h2.append(split_to_float(df[4]))
                    event_labels.append(np.array(labels_tmp, dtype=np.int32))
                    clusters = []
                    labels_tmp = []
                    del df
                items = event_line.split("|")[:-1]
                added_items = [c.split(";")[:-1] for c in items if c[:2] != "88"]
                clusters += added_items
                labels_tmp += [i] * len(added_items)
        df = pd.DataFrame(clusters)
        q1.append(split_to_float(df[0]))
        q2.append(split_to_float(df[1]))
        c.append(split_to_float(df[2]))
        h1.append(split_to_float(df[3]))
        h2.append(split_to_float(df[4]))
        event_labels.append(np.array(labels_tmp, dtype=np.int32))
        del clusters
        del labels_tmp
        del df
        q1 = np.concatenate(q1, dtype=np.float32, axis=0)
        q2 = np.concatenate(q2, dtype=np.float32, axis=0)
        c = np.concatenate(c, dtype=np.float32, axis=0)
        h1 = np.concatenate(h1, dtype=np.float32, axis=0)
        h2 = np.concatenate(h2, dtype=np.float32, axis=0)
        event_labels = np.concatenate(event_labels, dtype=np.int32, axis=0)
        return c, event_labels, h1, h2, q1, q2

    def summarize(self):
        print(f"Reading data from: {self.hparams.data_dir}")
        print(f"\tNumber of hadron types: {self.num_hadron_types}")
        print(f"\tNumber of conditional variables: {self.cond_dim}")
        print(f"\tNumber of output variables: {self.output_dim}")
        print(f"\tNumber of output hadrons: {self.hparams.num_output_hadrons}")
        print(
            f"\tNumber of particle kinematics: {self.hparams.num_particle_kinematics}"
        )


class HerwigClusterDataset(TorchDataset, HyperparametersMixin):
    """
    This module reads the original cluster decay files produced by Rivet,
    and it boosts the decay prodcuts to the cluster frame.
    Save the output to a new file for training the model.

    I define 3 cluster decay modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1

    Args:
        mode: the mode of the dataset
        with_quark (bool): two quark angles in the CM frame as inputs
        with_pert (bool): the `pertinent` flag of the two quarks as inputs
        with_evts (bool): one row containing one event, instead of one cluster
    """

    def __init__(
        self,
        root,
        mode: int,
        with_quark: bool,
        with_pert: bool,
        with_evts: bool,
        with_particle_type: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        pids_map_path = (
            os.path.join(root, pid_map_fname) if root is not None else pid_map_fname
        )
        if os.path.exists(pids_map_path):
            print("Loading existing pids map: ", pids_map_path)
            self.pids_to_ix = pickle.load(open(pids_map_path, "rb"))
        else:
            raise RuntimeError("No pids map found at", pids_map_path)

    def __len__(self):
        return self.num_evts

    def __getitem__(self, idx):
        pass

    def __call__(self, *args, **kwargs):
        if self.hparams.with_evts:
            self.convert_events(*args, **kwargs)
        else:
            self.convert_cluster_decay(*args, **kwargs)

    def convert_cluster_decay(
        self, filename: str, outname: str, do_check_only: bool = False
    ):
        mode = self.hparams.mode
        with_pert = self.hparams.with_pert
        with_quark = self.hparams.with_quark

        outname = (
            outname + f"_mode{mode}" + "_with_quark"
            if with_quark
            else outname + f"_mode{mode}"
        )
        outname = outname + "_with_pert" if with_pert else outname
        if do_check_only:
            self.check_converted_data(outname)
            return

        print(f"reading from {filename}")
        df = read_dataframe(filename, ";", "python")

        q1, q2, c, h1, h2 = [split_to_float(df[idx]) for idx in range(5)]

        if mode == 0:
            selections = (q1[5] == 1) & (q2[5] == 1)
            print("mode 0: both q1, q2 are with Pert=1")
        elif mode == 1:
            selections = ((q1[5] == 1) & (q2[5] == 0)) | ((q1[5] == 0) & (q2[5] == 1))
            print("mode 1: only one of q1 and q2 is with Pert=1")
        elif mode == 2:
            selections = (q1[5] == 0) & (q2[5] == 0)
            print("mode 2: neither q1 nor q2 are with Pert=1")
        elif mode == 3:
            selections = ~(q1[5] == 0) & (q2[5] == 0)
            print("mode 3: at least one quark with Pert=1")
        else:
            # no selections
            selections = slice(None)
            print(f"mode {mode} is not known! We will use all events.")

        cluster = c[[1, 2, 3, 4]][selections].values

        h1_types = h1[[0]][selections]
        h2_types = h2[[0]][selections]
        h1 = h1[[1, 2, 3, 4]][selections]
        h2 = h2[[1, 2, 3, 4]][selections]

        # to tell if the quark info is perserved to hadrons
        pert1 = q1[5][selections]
        pert2 = q2[5][selections]

        q1 = q1[[1, 2, 3, 4]][selections]
        q2 = q2[[1, 2, 3, 4]][selections]

        if with_quark:
            org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
        else:
            org_inputs = np.concatenate([cluster, h1, h2], axis=1)

        new_inputs = boost(org_inputs)

        out_4vec = new_inputs[:, -4:]
        _, px, py, pz = [out_4vec[:, idx] for idx in range(4)]
        pT = np.sqrt(px**2 + py**2)
        phi = np.arctan(px / py)
        theta = np.arctan(pT / pz)
        phi, theta = get_angles(new_inputs[:, -4:])

        out_truth = np.stack([phi, theta], axis=1)
        cond_info = cluster
        if with_quark:
            print("add quark information")
            # <NOTE, assuming the two quarks are back-to-back, xju>
            q_phi, q_theta = get_angles(new_inputs[:, 4:8])
            quark_angles = np.stack([q_phi, q_theta], axis=1)
            cond_info = np.concatenate([cond_info, quark_angles], axis=1)

        if with_pert:
            print("add pert information")
            pert_inputs = np.stack([pert1, pert2], axis=1)
            cond_info = np.concatenate([cond_info, pert_inputs], axis=1)

        scaler = InputScaler()

        # cond_info: conditional information
        # out_truth: the output hadron angles
        cond_info = scaler.transform(cond_info, outname + "_scalar_input4vec.pkl")
        out_truth = scaler.transform(out_truth, outname + "_scalar_outtruth.pkl")

        # add hadron types to the output, [phi, theta, type1, type2]
        out_truth = np.concatenate([out_truth, h1_types, h2_types], axis=1)
        np.savez(outname, cond_info=cond_info, out_truth=out_truth)

    def check_converted_data(self, outname):
        import matplotlib.pyplot as plt

        arrays = np.load(outname + ".npz")
        truth_in = arrays["out_truth"]
        plt.hist(truth_in[:, 0], bins=100, histtype="step", label="phi")
        plt.hist(truth_in[:, 1], bins=100, histtype="step", label="theta")
        plt.savefig("angles.png")

        scaler_input = InputScaler().load(outname + "_scalar_input4vec.pkl")
        scaler_output = InputScaler().load(outname + "_scalar_outtruth.pkl")

        print("//---- inputs ----")
        scaler_input.dump()
        print("//---- output ----")
        scaler_output.dump()
        print("Total entries:", truth_in.shape[0])

    def get_outname(self, outname):
        outname += (
            f"_mode{self.hparams.mode}"
            + ("_with_quark" if self.hparams.with_quark else "")
            + ("_with_pert" if self.hparams.with_pert else "")
        )
        return outname

    def load_pid_map(self):
        self.pids_map_fname = os.path.join(self.hparams.data_dir, pid_map_fname)
        if os.path.exists(self.pids_map_fname):
            print("Loading existing pids map: ", self.pids_map_fname)
            self.pids_to_ix = pickle.load(open(self.pids_map_fname, "rb"))
        else:
            raise RuntimeError("No pids map found")

    def convert_events(self, filename, outname, *args, **kwargs):
        with_quark = self.hparams.with_quark

        outname = self.get_outname(outname)
        if self.pids_to_ix is None:
            self.load_pid_map()

        print(f"reading from {filename}")
        datasets = []
        with open(filename) as f:
            for line in f:
                items = line.split("|")[:-1]
                clusters = [c.split(";")[:-1] for c in items]

                df = pd.DataFrame(clusters)
                q1, q2, c, h1, h2 = [split_to_float(df[idx]) for idx in range(5)]

                cluster = c[[1, 2, 3, 4]].values

                h1_types = h1[[0]]
                h2_types = h2[[0]]
                h1 = h1[[1, 2, 3, 4]]
                h2 = h2[[1, 2, 3, 4]]

                q1 = q1[[1, 2, 3, 4]]
                q2 = q2[[1, 2, 3, 4]]

                if with_quark:
                    org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
                else:
                    org_inputs = np.concatenate([cluster, h1, h2], axis=1)

                new_inputs = np.array([boost(row) for row in org_inputs])

                out_4vec = new_inputs[:, -4:]
                _, px, py, pz = [out_4vec[:, idx] for idx in range(4)]
                pT = np.sqrt(px**2 + py**2)
                phi = np.arctan(px / py)
                theta = np.arctan(pT / pz)
                phi, theta = get_angles(new_inputs[:, -4:])

                out_truth = np.stack([phi, theta], axis=1)
                cond_info = cluster

                # convert particle IDs to indices
                # then these indices can be embedded in N dim. space
                h1_type_indices = torch.from_numpy(
                    np.vectorize(self.pids_to_ix.get)(h1_types)
                )
                h2_type_indices = torch.from_numpy(
                    np.vectorize(self.pids_to_ix.get)(h2_types)
                )

                data = Data(
                    x=torch.from_numpy(out_truth),
                    edge_index=None,
                    cond_info=torch.from_numpy(cond_info),
                    ptypes=torch.from_numpy(
                        np.concatenate([h1_type_indices, h2_type_indices], axis=1)
                    ),
                )
                torch.save(
                    data,
                    os.path.join(
                        self.hparams.data_dir, f"{outname}_{len(datasets)}.pt"
                    ),
                )


class HerwigEventDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        raw_file_list=None,
        processed_file_name="herwig_graph_data.pt",
    ):

        self.raw_file_list = []
        for pattern in raw_file_list:
            self.raw_file_list += glob.glob(os.path.join(root, "raw", pattern))
        self.raw_file_list = [
            os.path.basename(raw_file) for raw_file in self.raw_file_list
        ]
        self.processed_file_name = processed_file_name

        if root:
            pids_map_path = os.path.join(root, pid_map_fname)
        if os.path.exists(pids_map_path):
            print("Loading existing pids map: ", pids_map_path)
            self.pids_to_ix = pickle.load(open(pids_map_path, "rb"))
        else:
            raise RuntimeError("No pids map found at", pids_map_path)

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.raw_file_list is not None:
            return self.raw_file_list
        return ["ClusterTo2Pi0_new.dat"]

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def download(self):
        pass

    def process(self):
        all_data = []
        for raw_path in self.raw_paths:
            with open(raw_path) as f:
                data_list = [self._create_data(line) for line in f]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            all_data += data_list

        data, slices = self.collate(all_data)
        torch.save((data, slices), self.processed_paths[0])

    def _create_data(self, line):
        with_quark = False

        items = line.split("|")[:-1]
        clusters = [c.split(";")[:-1] for c in items]

        df = pd.DataFrame(clusters)
        q1, q2, c, h1, h2 = [split_to_float(df[idx]) for idx in range(5)]

        cluster = c[[1, 2, 3, 4]].values

        h1_types = h1[[0]]
        h2_types = h2[[0]]
        h1 = h1[[1, 2, 3, 4]]
        h2 = h2[[1, 2, 3, 4]]

        q1 = q1[[1, 2, 3, 4]]
        q2 = q2[[1, 2, 3, 4]]

        if with_quark:
            org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
        else:
            org_inputs = np.concatenate([cluster, h1, h2], axis=1)

        new_inputs = boost(org_inputs)

        def get_angles(four_vector):
            _, px, py, pz = [four_vector[:, idx] for idx in range(4)]
            pT = np.sqrt(px**2 + py**2)
            phi = np.arctan(px / py)
            theta = np.arctan(pT / pz)
            return phi, theta

        phi, theta = get_angles(new_inputs[:, -4:])
        theta = theta + math.pi * (theta < 0)

        angles = np.stack([phi, theta], axis=1)
        hadrons = np.concatenate([h1, h2], axis=1)

        # convert particle IDs to indices
        # then these indices can be embedded in N dim. space
        h1_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h1_types))
        h2_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h2_types))

        data = Data(
            x=torch.from_numpy(angles).float(),
            hadrons=torch.from_numpy(hadrons).float(),
            edge_index=None,
            cluster=torch.from_numpy(cluster).float(),
            ptypes=torch.from_numpy(
                np.concatenate([h1_type_indices, h2_type_indices], axis=1)
            ).long(),
        )
        return data


class HerwigMultiHadronEventDataset(Dataset):
    """ This class takes preprocessed multihadron events (i.e. clusters and multiple hadrons having 
    different types transformed to indices) and provide the DataLoader class object with prepared 
    generator and discriminator sentences. """

    def __init__(self, n_had_types, clusters, hadrons_with_types, max_n_hadrons):
        super().__init__()

        self.n_had_types = n_had_types
        self.hadrons_with_types = hadrons_with_types
        self.clusters = clusters
        self.n_clusters = len(clusters)
        self.max_n_hadrons = max_n_hadrons
        # Cluster padding token (zeros)
        self.cluster_padding_token = torch.zeros(1, len(clusters[0]))
        # Hadron padding token (zeros + special hadron type: the first one-hot position)
        self.hadron_padding_token = torch.zeros(1, len(hadrons_with_types[0][0][0]) + n_had_types)
        self.hadron_padding_token[0, len(hadrons_with_types[0][0][0])] = 1.0


    def __len__(self):
        return self.n_clusters


    def __getitem__(self, index):
        """ This method is mainly responsible for tokenisation and building sentences containing 
        "cluster/hadron padding tokens". It returns a pair of prepared sentences. """

        # Preparing the input sentence. Tokens then need to be concatenated with noise.
        # [[cluster_kin][cluster_kin]...[padding_token]] 
        # N = max number of hadrons produced by the heaviest cluster
        n_hadrons = len(self.hadrons_with_types[index][0])
        gen_input = torch.cat([self.clusters[index].unsqueeze(0) for _ in range(n_hadrons)])
        if self.max_n_hadrons - n_hadrons > 0:
            padding_tokens = torch.cat([self.cluster_padding_token for 
                                        _ in range(self.max_n_hadrons - n_hadrons)])
            gen_input = torch.cat([gen_input, padding_tokens])
        
        # Preparing the output sentence. The hadron type is a one-hot vector.
        # [[hadron_kin, hadron_type][hadron_kin, hadron_type]...[padding_token]]
        # N = max number of hadrons produced by the heaviest cluster
        hadrons = self.hadrons_with_types[index][0]
        hadron_types = self.hadrons_with_types[index][1]
        hadron_types_ohe = torch.nn.functional.one_hot(hadron_types, self.n_had_types)
        disc_input = torch.cat([hadrons, hadron_types_ohe], dim=1)
        if self.max_n_hadrons - n_hadrons > 0:
            hadron_padding_tokens = torch.cat([self.hadron_padding_token 
                                            for _ in range(self.max_n_hadrons - n_hadrons)])
            disc_input = torch.cat([disc_input, hadron_padding_tokens])
        
        return gen_input, disc_input