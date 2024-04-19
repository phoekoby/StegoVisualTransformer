from functools import partial
from typing import Callable

import torch
from torch import nn
from torchvision.models.vision_transformer import MLPBlock

from model_utils import create_patches

from collections import OrderedDict


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            in_channels,
            hidden_dim,
            dropout,
            device
    ):
        super(PatchEmbedding, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        n_patches = (image_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout)

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.pos_embedding = nn.Parameter(torch.randn(size=(1, n_patches + 1, hidden_dim), requires_grad=True))


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv1(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        x = self._process_input(x)

        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)

        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class Img2Seq(nn.Module):
    def __init__(self, img_size, patch_size, n_channels, hidden_dim, device):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.device = device

        height, width = img_size

        self.patches_amount = ((height // patch_size[0]) * (width // patch_size[1]))

        token_dim = patch_size[0] * patch_size[1] * n_channels
        self.linear = nn.Linear(token_dim, hidden_dim)
        self.class_token = nn.Parameter(torch.rand(1, hidden_dim))

        # self.pos_embed = nn.Parameter(get_positional_embeddings(self.patches_amount + 1, hidden_dim).clone().detach().requires_grad_(False))
        self.pos_embed = nn.Parameter(torch.empty(1, self.patches_amount + 1, hidden_dim).normal_(std=0.02))  # from BERT

    def __call__(self, batch):
        n, c, h, w = batch.shape

        patches = create_patches(batch, self.patch_size).to(self.device)

        # nn.Embedding

        tokens = self.linear(patches)

        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        pos_embed = self.pos_embed.repeat(n, 1, 1)

        out = tokens + pos_embed

        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
