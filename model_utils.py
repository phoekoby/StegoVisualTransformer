import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor


def create_patches(input, patch_size):
    amount, channels, height, width = input.shape

    amount_of_patches = ((height // patch_size[0]) * (width // patch_size[1]))

    patches = torch.zeros(amount, amount_of_patches, channels * height * width // amount_of_patches)
    for idx, image in enumerate(input):
        for i in range(height // patch_size[0]):
            for j in range(width // patch_size[1]):
                patch = image[:, i * patch_size[0]: (i + 1) * patch_size[0], j * patch_size[0]: (j + 1) * patch_size[1]]
                patches[idx, i * (height // patch_size[0]) + j] = patch.flatten()

    return patches


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                        :,
                        i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size,
                        ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
            return result


def get_mlp(in_features, hidden_units, out_features):
    """
    Returns a MLP head
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, 1, hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)