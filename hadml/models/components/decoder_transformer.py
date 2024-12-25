import torch.nn as nn


class DecoderTransformer(nn.Transformer):
    """ Template for a decoder-only transformer model """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x