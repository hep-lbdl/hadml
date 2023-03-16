"""Multilayer Perceptron (MLP) module."""
from typing import List, Optional, Tuple

try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_linear_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    layer_norm: bool = True,
    dropout: float = 0.0,
    last_activation: Optional[torch.nn.Module] = None,
    leaky_ratio: float = 0.2,
) -> List[nn.Module]:

    layer_list = [
        torch.nn.Linear(input_dim, hidden_dims[0]),
        torch.nn.LeakyReLU(leaky_ratio),
    ]

    for l0, l1 in pairwise(hidden_dims):
        layer_list.append(torch.nn.Linear(l0, l1))

        if layer_norm:
            layer_list.append(torch.nn.LayerNorm(l1))

        layer_list.append(torch.nn.LeakyReLU(leaky_ratio))

        if dropout > 0:
            layer_list.append(torch.nn.Dropout(dropout))

    layer_list.append(torch.nn.Linear(hidden_dims[-1], output_dim))
    if last_activation is not None:
        layer_list.append(last_activation)

    return layer_list


class MLPModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        layer_norm: bool = True,
        dropout: float = 0.0,
        last_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        # build the linear model
        self.model = nn.Sequential(
            *build_linear_layers(
                input_dim, hidden_dims, output_dim, layer_norm, dropout, last_activation
            )
        )

    def forward(self, x, *args) -> torch.Tensor:
        if len(args) > 0:
            x = torch.cat((x,) + args, dim=1)
        return self.model(x)

class MLPParticleModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kinematic_dim: int,
        particle_type_dim: int,
        num_output_particles: int,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # build the linear model
        self.encoder = nn.Sequential(
            *build_linear_layers(
                input_dim, hidden_dims, hidden_dims[-1], layer_norm, dropout
            )
        )
        self.particle_type = nn.Sequential(
            *build_linear_layers(
                hidden_dims[-1],
                [particle_type_dim * 2],
                particle_type_dim,
                layer_norm,
                dropout,
                torch.nn.LogSoftmax(dim=-1),
            )
        )

        self.particle_type = nn.Sequential(*build_linear_layers(
            int(hidden_dims[-1] / num_output_particles), [particle_type_dim*2], particle_type_dim, layer_norm, dropout, torch.nn.LogSoftmax(dim=-1)))
                                           
        self.particle_kinematics = nn.Sequential(*build_linear_layers(
            hidden_dims[-1], [kinematic_dim*2], kinematic_dim, layer_norm, dropout, torch.nn.Tanh()))
        
        self.num_output_particles = num_output_particles                                            
                                           
        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        encode = self.encoder(x)
        p_kines = self.particle_kinematics(encode)

        type_encode = encode.reshape(batch_size * self.num_output_particles, -1)
        p_types = self.particle_type(type_encode)

        return p_kines, p_types


class OneHotEmbeddingModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
    ):
        super().__init__()
        self.num_classes = vocab_size

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(-1)
        embeds = F.one_hot(x, self.num_classes).view(batch_size, -1)
        return embeds


class MLPOneHotEmbeddingModule(OneHotEmbeddingModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(vocab_size=vocab_size)

        # build the linear model
        self.model = nn.Sequential(
            *build_linear_layers(vocab_size, hidden_dims, output_dim, True, dropout)
        )

    def forward(self, x) -> torch.Tensor:
        embeds = super().forward(x)
        return self.model(embeds)


class MLPTypeEmbeddingModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # build the linear model
        self.model = nn.Sequential(
            *build_linear_layers(embedding_dim, hidden_dims, output_dim, True, dropout)
        )

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(-1)
        embeds = self.embeddings(x).view(batch_size, -1)
        return self.model(embeds)


class MLPWithEmbeddingModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        word_embedding_dim: int,  # only used if embedding_type is dense
        num_words: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        output_dim: int,
        last_activation: Optional[torch.nn.Module] = None,
        dropout: float = 0.0,
        embedding_type: str = "onehot",  # onehot or dense
    ):
        super().__init__()

        self.normal_mlp = MLPModule(input_dim, encoder_dims, encoder_dims[-1])

        if embedding_type == "onehot":
            self.type_mlp = MLPOneHotEmbeddingModule(
                vocab_size, encoder_dims, encoder_dims[-1], dropout=dropout
            )
        elif embedding_type == "dense":
            self.type_mlp = MLPTypeEmbeddingModule(
                vocab_size,
                word_embedding_dim,
                encoder_dims,
                encoder_dims[-1],
                dropout=dropout,
            )
        else:
            raise ValueError(f"embedding_type {embedding_type} not supported")

        self.decoder = MLPModule(
            encoder_dims[-1] * (1 + num_words),
            decoder_dims,
            output_dim,
            last_activation=last_activation,
        )

    def forward(self, x, type_ids) -> torch.Tensor:
        normal_embeds = self.normal_mlp(x)
        num_particles = type_ids.shape[1]

        # same MLP acting on different particles
        type_embeds = [self.type_mlp(type_ids[:, i]) for i in range(num_particles)]

        decoder_embds = self.decoder(torch.cat([normal_embeds] + type_embeds, dim=1))
        return decoder_embds


if __name__ == "__main__":
    model = MLPModule(784, [256, 256, 256], 10)
    print(model)
