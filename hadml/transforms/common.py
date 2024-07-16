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
    def __init__(self, file, num_particles) -> None:
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

class CombinedEmbedder(BaseTransform):
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
        oh = torch.eye(self.num_particles)
        embeddings = np.concatenate([embeddings, oh], axis=-1)
        self.embedding_dim = embeddings.shape[1]
        self.embedding = torch.from_numpy(embeddings)

    def apply_func(self, x: torch.Tensor):
        embedded = self.embedding[x.flatten().long()].reshape(*x.shape, self.embedding_dim)
        return embedded

    def __repr__(self) -> str:
        return super().__repr__() + f"(file={self.file})"

selected_pids = [143, 142, 147, 148, 144, 141, 152, 151, 159, 134,] # 155, 138, 133, 160, 156, 137, 109, 188, 111, 186, 153, 240, 149, 140, 145, 234, 63, 237, 277, 278, 29, 279, 318, 0, 317, 319, 154, 310, 150, 146, 139, 60, 246, 57, 243, 5, 308, 241, 306, 132, 161, 136, 157, 62, 238, 235, 163, 104, 193, 164, 126, 108, 171, 189, 168, 129, 110, 187, 282, 26, 28, 280, 112, 194, 107, 103, 185, 314, 190, 312, 3, 1, 196, 101, 117, 177, 120, 180, 309, 191, 6, 106, 307, 305, 127, 170, 130, 167, 301, 42, 263, 264, 43, 300, 299, 10, 162, 131, 158, 239, 135, 181, 119, 178, 116, 292, 17, 291, 18, 236, 64, 233, 59, 247, 244, 56, 320, 25, 281, 27, 283, 179, 118, 176, 121, 258, 46, 48, 260, 165, 288, 102, 195, 21, 289, 9, 20, 8, 303, 302]

class OneHot_Emb_Concat(BaseTransform):
    """
    This method combines One Hot along with a specified embedding. It concatenates the two, hoping the benefits of both can be used
    """
    def __init__(self, num_particles) -> None:
        super().__init__()
        self.num_particles = num_particles
        mini_parts = 10
        self.mini_parts = mini_parts
        temp = torch.eye(mini_parts)
        self.embedding = torch.zeros((self.num_particles, mini_parts))
        for i in range(len(selected_pids)):
            self.embedding[selected_pids[i]] = temp[i]

    def apply_func(self, x: torch.Tensor):
        embedded = self.embedding[x.flatten().long()].reshape(*x.shape, self.mini_parts)
        return embedded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_particles={self.num_particles})"


class PID_Encode(BaseTransform):
    """
    One-Hot Encodde the pids
    This method expects indices as input
    """
    def __init__(self, file="data/Herwig/pids_to_ix.pkl", num_particles) -> None:
        super().__init__()
        self.num_particles = num_particles
        self.file = file
        import pickle as pkl
        self.embedding_conv = pkl.load(self.file)

    def apply_func(self, x: torch.Tensor):
        embedded = self.embedding[idx.item() for idx in x]
        return torch.tensor(embedded)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_particles={self.num_particles})"