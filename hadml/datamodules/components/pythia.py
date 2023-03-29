import os
import pickle
import glob

import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

pid_map_fname = "pids_to_ix.pkl"


class PythiaEventDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, raw_file_list=None,
                 processed_file_name='pythia_graph_data.pt'):

        self.raw_file_list = []
        for pattern in raw_file_list:
            self.raw_file_list += glob.glob(os.path.join(root, "raw", pattern))
        self.raw_file_list = [
            os.path.basename(raw_file) for raw_file in self.raw_file_list]
        self.processed_file_name = processed_file_name

        if root:
            pids_map_path = os.path.join(root, pid_map_fname)
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
        return ['pythia_data.dat']

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def download(self):
        pass

    def process(self):
        all_data = []
        for raw_path in self.raw_paths:
            with open(raw_path) as f:
                data_list = [self._create_data(line) for line in f][:-1]

            if self.pre_filter is not None:
                data_list = [
                    data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            all_data += data_list

        data, slices = self.collate(all_data)
        torch.save((data, slices), self.processed_paths[0])

    def _create_data(self, line):

        particles = line.split(';')[:-1]
        particles = [particle.split(',') for particle in particles]

        particles = np.array(particles).astype(float)

        if particles.shape == (0,):
            return

        h_types = particles[:, 0:1]
        h = particles[:, 1:]

        # convert particle IDs to indices
        # then these indices can be embedded in N dim. space
        h_type_indices = torch.from_numpy(
            np.vectorize(self.pids_to_ix.get)(h_types))

        data = Data(
            x=torch.from_numpy(h).float(),
            hadrons=torch.from_numpy(h).float(),
            edge_index=None,
            cluster=None,
            ptypes=h_type_indices.long(),
        )
        return data
