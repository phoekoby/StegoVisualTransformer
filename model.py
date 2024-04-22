import random

import numpy as np
import torch
from torch import nn

from layers import PatchEmbedding, Encoder
from model_utils import get_mlp
import torch.nn.functional as F


class KovViT(nn.Module):
    def __init__(
            self,
            device=None,
            batch_size=32,
            img_size=256,
            patch_size=8,
            n_channels=1,
            hidden_dim=1024,
            nhead=32,
            num_layers=2,
            mlp_dim=1024,
            n_classes=2,
            dropout=0.1,
            emb_dropout=0.1,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        # self.n_channels = 30
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.srm = SRMConv(device=device)

        self.patch_embedding = PatchEmbedding(
            image_size=img_size,
            patch_size=self.patch_size,
            in_channels=self.n_channels,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            device=device
        )

        # self.img2seq = Img2Seq(self.img_size, self.patch_size, self.n_channels, self.hidden_dim, device=device)
        self.encoder = Encoder(
            seq_length=(self.img_size // self.patch_size) ** 2 + 1,
            num_layers=self.num_layers,
            num_heads=self.nhead,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
            attention_dropout=emb_dropout
        )
        # encoder_layer = nn.TransformerEncoderLayer(
        #     self.hidden_dim, self.nhead, self.dim_feedforward, activation="elu", batch_first=True
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer, self.num_layers
        # )
        #
        # self.mlp = get_mlp(self.hidden_dim,  self.mlp_head_units, self.n_classes)

        self.output = nn.Softmax(dim=-1)

    def forward(self, inp):
        # out = self.srm(inp)
        out = self.patch_embedding(inp)

        out = self.encoder(out)

        out = out[:, 0]

        # out = self.mlp(out)

        out = self.output(out)

        return out


class SRMConv(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.srm = torch.from_numpy(np.load("srm.npy")).to(device).type(torch.cuda.FloatTensor)
        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, inp):
        return self.tlu(
            F.conv2d(
                inp,
                self.srm,
                stride=(1, 1),
                padding=2,
                bias=torch.from_numpy(np.ones(30)).type(torch.cuda.FloatTensor))
        )


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.query = nn.Linear(dim, dim_head * heads)
        self.key = nn.Linear(dim, dim_head * heads)
        self.value = nn.Linear(dim, dim_head * heads)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            q, k, v = self.query(x), self.key(x), self.value(x)
            w, attention = attn(q, k, v)
            x = w + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., device=None):

        super().__init__()

        self.srm = SRMConv(device=device)
        channels=30

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        img = self.srm(img)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)