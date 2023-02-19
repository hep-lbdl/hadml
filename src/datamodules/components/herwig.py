
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.mixins import HyperparametersMixin

from src.datamodules.components.utils import (
    read_dataframe, split_to_float, InputScaler, boost
)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset


pid_map_fname = "pids_to_ix.pkl"

class Herwig(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "data/",
        fname: str = "allHadrons_10M_mode4_with_quark_with_pert.npz",
        original_fname: str = "cluster_ML_allHadrons_10M.txt",
        train_val_test_split: Tuple[int, int, int] = (100, 50, 50),
        num_output_hadrons: int = 2,
        num_particle_kinematics: int = 2,
        # hadron_type_embedding_dim: int = 10,
    ):
        """This is for the GAN datamodule. It reads clusters from a file"""
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.cond_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        self.pids_to_ix: Optional[Dict[int, int]] = None
        
        ## particle type map
        self.pids_map_fname = os.path.join(self.hparams.data_dir, pid_map_fname)
        self.num_hadron_types: int = 0
    
    
    def prepare_data(self):
        ## read the original file, determine the number of particle types
        ## and create a map.
        if os.path.exists(self.pids_map_fname):
            print("Loading existing pids map")
            self.pids_to_ix = pickle.load(open(self.pids_map_fname, 'rb'))
            self.num_hadron_types = len(list(self.pids_to_ix.keys()))
            print("END...Loading existing pids map")
        else:
            fname = os.path.join(self.hparams.data_dir, self.hparams.origin_fname)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} not found.")
            df = pd.read_csv(fname, sep=';', header=None, names=None, engine='python')
            
            def split_to_float(df, sep=','):
                out = df
                if type(df.iloc[0]) == str:
                    out = df.str.split(sep, expand=True).astype(np.float32)
                return out
            
            q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]
            h1_type, h2_type = h1[[0]], h2[[0]]
            hadron_pids = np.unique(np.concatenate([h1_type, h2_type])).astype(np.int64)
            
            self.pids_to_ix = {pids: i for i, pids in enumerate(hadron_pids)}
            self.num_hadron_types = len(hadron_pids)
            
            pickle.dump(self.pids_to_ix, open(self.pids_map_fname, "wb"))
            
            
    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"It creates the dataset for training a conditional GAN.
        Returns:
            cond_info: conditional information
            x_truth:   target truth information with conditonal information
        """
        fname = os.path.join(self.hparams.data_dir, self.hparams.fname)
        arrays = np.load(fname)
        
        cond_info = torch.from_numpy(arrays['cond_info'].astype(np.float32))
        truth_in = torch.from_numpy(arrays['out_truth'].astype(np.float32))
        
        num_tot_evts, self.cond_dim = cond_info.shape
        num_asked_evts = sum(self.hparams.train_val_test_split)
        
        print(f"Number of events: {num_tot_evts:,}, asking for {num_asked_evts:,}")
        if num_tot_evts < num_asked_evts:
            raise ValueError(f"Number of events {num_tot_evts} is less than asked {num_asked_evts}")
        
        cond_info = cond_info[:num_asked_evts]
        truth_in = truth_in[:num_asked_evts]
        
        ## output includes N hadron types and their momenta
        ## output dimension only includes the momenta
        self.output_dim = truth_in.shape[1] - self.hparams.num_output_hadrons

        true_hadron_momenta = truth_in[:, :-self.hparams.num_output_hadrons]
           
        ## convert particle IDs to indices
        ## then these indices can be embedded in N dim. space
        target_hadron_types = truth_in[:, -self.hparams.num_output_hadrons:].reshape(-1).long()
        target_hadron_types_idx = torch.from_numpy(np.vectorize(
            self.pids_to_ix.get)(target_hadron_types.numpy())).reshape(-1, self.hparams.num_output_hadrons)
        
        self.summarize()
        return (cond_info, true_hadron_momenta, target_hadron_types_idx)
    
    
    def summarize(self):
        print(f"Reading data from: {self.hparams.data_dir}")
        print(f"\tNumber of hadron types: {self.num_hadron_types}")
        print(f"\tNumber of conditional variables: {self.cond_dim}")
        print(f"\tNumber of output variables: {self.output_dim}")
        print(f"\tNumber of output hadrons: {self.hparams.num_output_hadrons}")
        print(f"\tNumber of particle kinematics: {self.hparams.num_particle_kinematics}")
        
        
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
        with_quark (bool): add two quark angles in the center-of-mass frame as inputs
        with_pert (bool): add the `pertinent` flag of the two quarks as inputs
        with_evts (bool): one row containing one event, instead of one cluster...
    """
    def __init__(
        self, root,
        mode: int,
        with_quark: bool,
        with_pert: bool,
        with_evts: bool,
        with_particle_type: bool,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        pids_map_path = os.path.join(root, pid_map_fname) if root is not None else pid_map_fname
        if os.path.exists(pids_map_path):
            print("Loading existing pids map: ", pids_map_path)
            self.pids_to_ix = pickle.load(open(pids_map_path, 'rb'))
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

    def convert_cluster_decay(self, filename: str, outname: str, do_check_only: bool = False):
        mode = self.hparams.mode
        with_pert = self.hparams.with_pert
        with_quark = self.hparams.with_quark
        
        outname = outname+f"_mode{mode}"+"_with_quark" if with_quark else outname+f"_mode{mode}"
        outname = outname+"_with_pert" if with_pert else outname
        if do_check_only:
            self.check_converted_data(outname)
            return

        print(f'reading from {filename}')
        df = read_dataframe(filename, ";", 'python')

        q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]

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
            ## no selections
            selections = slice(None)
            print(f"mode {mode} is not known! We will use all events.")

        cluster = c[[1, 2, 3, 4]][selections].values
        
        h1_types = h1[[0]][selections]
        h2_types = h2[[0]][selections]
        h1 = h1[[1, 2, 3, 4]][selections]
        h2 = h2[[1, 2, 3, 4]][selections]


        ## to tell if the quark info is perserved to hadrons
        pert1 = q1[5][selections]
        pert2 = q2[5][selections]

        q1 = q1[[1, 2, 3, 4]][selections]
        q2 = q2[[1, 2, 3, 4]][selections]

        if with_quark:
            org_inputs = np.concatenate([cluster, q1, q2, h1, h2], axis=1)
        else:
            org_inputs = np.concatenate([cluster, h1, h2], axis=1)


        new_inputs = np.array([boost(row) for row in org_inputs])

        def get_angles(four_vector):
            _,px,py,pz = [four_vector[:, idx] for idx in range(4)]
            pT = np.sqrt(px**2 + py**2)
            phi = np.arctan(px/py)
            theta = np.arctan(pT/pz)
            return phi, theta

        out_4vec = new_inputs[:, -4:]
        _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
        pT = np.sqrt(px**2 + py**2)
        phi = np.arctan(px/py)
        theta = np.arctan(pT/pz)
        phi, theta = get_angles(new_inputs[:, -4:])
        
        out_truth = np.stack([phi, theta], axis=1)
        cond_info = cluster
        if with_quark:
            print("add quark information")
            ## <NOTE, assuming the two quarks are back-to-back, xju>
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
        cond_info = scaler.transform(cond_info, outname+"_scalar_input4vec.pkl")
        out_truth = scaler.transform(out_truth, outname+"_scalar_outtruth.pkl")
        
        # add hadron types to the output, [phi, theta, type1, type2]
        out_truth = np.concatenate([out_truth, h1_types, h2_types], axis=1)
        np.savez(outname, cond_info=cond_info, out_truth=out_truth)
    
    def check_converted_data(self, outname):
        import matplotlib.pyplot as plt

        arrays = np.load(outname+".npz")
        truth_in = arrays['out_truth']
        plt.hist(truth_in[:, 0], bins=100, histtype='step', label='phi')
        plt.hist(truth_in[:, 1], bins=100, histtype='step', label='theta')
        plt.savefig("angles.png")

        scaler_input = InputScaler().load(outname+"_scalar_input4vec.pkl")
        scaler_output = InputScaler().load(outname+"_scalar_outtruth.pkl")

        print("//---- inputs ----")
        scaler_input.dump()
        print("//---- output ----")
        scaler_output.dump()
        print("Total entries:", truth_in.shape[0])
    
    def get_outname(self, outname):
        outname = outname+f"_mode{self.hparams.mode}"+"_with_quark" if self.hparams.with_quark \
            else outname+f"_mode{self.hparams.mode}"
        outname = outname+"_with_pert" if self.hparams.with_pert else outname
        return outname
        
        
    def load_pid_map(self):
        self.pids_map_fname = os.path.join(self.hparams.data_dir, pid_map_fname)
        if os.path.exists(self.pids_map_fname):
            print("Loading existing pids map: ", self.pids_map_fname)
            self.pids_to_ix = pickle.load(open(self.pids_map_fname, 'rb'))
        else:
            raise RuntimeError("No pids map found")
            
            
    def convert_events(self, filename, outname, *args, **kwargs):
        with_quark = self.hparams.with_quark
        
        outname = self.get_outname(outname)
        if self.pids_to_ix is None:
            self.load_pid_map()
        
        print(f'reading from {filename}')
        datasets = []
        with open(filename) as f:
            for line in f:
                items = line.split('|')[:-1]
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

                def get_angles(four_vector):
                    _,px,py,pz = [four_vector[:, idx] for idx in range(4)]
                    pT = np.sqrt(px**2 + py**2)
                    phi = np.arctan(px/py)
                    theta = np.arctan(pT/pz)
                    return phi, theta

                out_4vec = new_inputs[:, -4:]
                _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
                pT = np.sqrt(px**2 + py**2)
                phi = np.arctan(px/py)
                theta = np.arctan(pT/pz)
                phi, theta = get_angles(new_inputs[:, -4:])
                    
                out_truth = np.stack([phi, theta], axis=1)
                cond_info = cluster
                
                ## convert particle IDs to indices
                ## then these indices can be embedded in N dim. space
                h1_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h1_types))
                h2_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h2_types))
        
                data = Data(
                    x=torch.from_numpy(out_truth),
                    edge_index=None,
                    cond_info=torch.from_numpy(cond_info),
                    ptypes=torch.from_numpy(np.concatenate([h1_type_indices, h2_type_indices], axis=1)),
                )
                torch.save(data, os.path.join(self.hparams.data_dir, outname+f"_{len(datasets)}.pt"))
                
                
class HerwigEventDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, raw_file_list=None):
        
        self.raw_file_list = raw_file_list
        
        pids_map_path = os.path.join(root, pid_map_fname) if root is not None else pid_map_fname
        if os.path.exists(pids_map_path):
            print("Loading existing pids map: ", pids_map_path)
            self.pids_to_ix = pickle.load(open(pids_map_path, 'rb'))
        else:
            raise RuntimeError("No pids map found at", pids_map_path)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        if self.raw_file_list is not None:
            return self.raw_file_list
        return ['ClusterTo2Pi0_large.dat']
    
    @property
    def processed_file_names(self):
        return ['herwig_graph_data.pt']
    
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
        
        items = line.split('|')[:-1]
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

        def get_angles(four_vector):
            _,px,py,pz = [four_vector[:, idx] for idx in range(4)]
            pT = np.sqrt(px**2 + py**2)
            phi = np.arctan(px/py)
            theta = np.arctan(pT/pz)
            return phi, theta

        out_4vec = new_inputs[:, -4:]
        _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
        pT = np.sqrt(px**2 + py**2)
        phi = np.arctan(px/py)
        theta = np.arctan(pT/pz)
        phi, theta = get_angles(new_inputs[:, -4:])
            
        out_truth = np.stack([phi, theta], axis=1)
        cond_info = cluster
        
        ## convert particle IDs to indices
        ## then these indices can be embedded in N dim. space
        h1_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h1_types))
        h2_type_indices = torch.from_numpy(np.vectorize(self.pids_to_ix.get)(h2_types))

        data = Data(
            x=torch.from_numpy(out_truth).float(),
            edge_index=None,
            cond_info=torch.from_numpy(cond_info).float(),
            ptypes=torch.from_numpy(np.concatenate([h1_type_indices, h2_type_indices], axis=1)).long(),
        )
        return data
    
    