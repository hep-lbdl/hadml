from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from torch_scatter import scatter_mean

from hadml.models.components.mlp import MLPModule


class DeepSetModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        output_dim: int,
        last_activation: Optional[nn.Module] = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.encoder = MLPModule(input_dim,
                                 encoder_dims,
                                 decoder_dims[0],
                                 layer_norm=layer_norm)
        self.decoder = MLPModule(
            decoder_dims[0],
            decoder_dims,
            output_dim,
            last_activation=last_activation,
            layer_norm=layer_norm
        )

    def forward(self, x: Tensor, batch: Tensor, *args, **kwargs) -> Tensor:
        # batch is a column vector which maps each node to its respective graph in the batch
        # https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/introduction.html#mini-batches

        encoded = self.encoder(x)
        # now summed_info: [num_evts, decoder_dims[0]]
        summed_info = scatter_mean(encoded, batch, dim=0)

        return self.decoder(summed_info)


class ClusterInEventModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        output_dim: int,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = MLPModule(input_dim, encoder_dims, decoder_dims[0])
        self.decoder = MLPModule(
            decoder_dims[0], decoder_dims, output_dim, last_activation=last_activation
        )

    def forward(self, x: Tensor, batch: Tensor, *args, **kwargs) -> Tensor:
        # batch is a column vector which maps each node to its respective graph in the batch
        # https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/introduction.html#mini-batches

        encoded = self.encoder(x)
        return self.decoder(encoded)
