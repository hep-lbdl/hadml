from torch import Tensor, nn


class ls(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score: Tensor, label: Tensor, *args, **kwargs) -> Tensor:
        ls = ((label) * ((score - 1) ** 2) + (1 - label) * (score**2)).mean()
        return ls


class wasserstein(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score: Tensor, label: Tensor, *args, **kwargs) -> Tensor:
        ls = (-label * score + (1 - label) * score).mean()
        return ls
