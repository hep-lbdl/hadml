"""
Would include one-hot encoding
"""

from typing import Any
import numpy as np
import torch
from torch_geometric.data import Data

REPLACE_FEATURE = 'ptypes'
class BaseTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, data: Data) -> Data:
        new_data = data.apply(self.apply_func, REPLACE_FEATURE)
        return new_data

    def apply_func(self, x: torch.Tensor):
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class OneHotEncode(BaseTransform):
    """
    One-Hot Encodde the pids
    This method expects indices as input
    """
    def __init__(self, num_particles) -> None:
        super().__init__()
        self.num_particles = num_particles
        self.embedding = torch.eye(self.num_particles)

    def apply_func(self, x: torch.Tensor):
        embedded = self.embedding[x.flatten().long()].reshape(*x.shape, self.num_particles)
        return embedded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_particles={self.num_particles})"


class IndexEmbedder(BaseTransform):
    def __init__(self, file, num_particles) -> None:
        super().__init__()
        self.file=file
        self.num_particles = num_particles
        self.load_embeddings()

    def load_embeddings(self):
        """Loads the embedding file
        Expected to be in npy format?
        """
        embeddings = np.load(self.file)
        assert embeddings.shape[0] == self.num_particles
        self.embedding_dim = embeddings.shape[1]
        self.embedding = torch.from_numpy(embeddings)

    def apply_func(self, x: torch.Tensor):
        embedded = self.embedding[x.flatten().long()].reshape(*x.shape, self.embedding_dim)
        return embedded

    def __repr__(self) -> str:
        return super().__repr__() + f"(file={self.file})"