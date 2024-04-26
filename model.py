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
        self.tlu = nn.Hardtanh(min_val=-5.0, max_val=5.0)

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

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class Transformer(nn.Module):
    def __init__(self, dim, depth, n_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.layers = nn.ModuleList([])

        # self.out = nn.Linear(dim_head * n_heads, dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
                MLP(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        batch_size, n_patch, _ = x.size()
        # (B, N, D) -> (B, nb_head, N, D//nb_head)
        for attn, ff in self.layers:
            q, k, v = self.query(x), self.key(x), self.value(x)
            # print(q.shape)

            # q = q.view(batch_size, self.n_heads, n_patch, self.dim_head)
            # k = k.view(batch_size, self.n_heads, n_patch, self.dim_head)
            # v = v.view(batch_size, self.n_heads, n_patch, self.dim_head)

            w, attention = attn(q, k, v)
            x = w + x
            x = ff(x) + x
        # x = self.out(x)
        return self.norm(x)


class MLPHead(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=32, num_classes=2, fine_tune=False):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes

        if not fine_tune:
            # hidden layer with tanh activation function
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),  # hidden layer
                nn.Tanh(),
                nn.Linear(hidden_dim, num_classes)  # output layer
            )
        else:
            # single linear layer
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class GaussianActivationLayer(nn.Module):
    def __init__(self, init_sigma):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        out = torch.exp(torch.div(torch.multiply(torch.pow(x, 2), -0.5), self.sigma ** 2))
        return out

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim_model, depth, heads, mlp_dim, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., device=None):
        super().__init__()

        # self.srm = SRMConv(device=device)
        # channels = 30

        # self.gauss = GaussianActivationLayer(0.1 * random.random())
        # Увеличивает стеганографический шум
        channels = 32
        self.eps = 0.001
        self.srm = nn.Sequential(
            SRMConv(device=device),
            nn.BatchNorm2d(30, eps=self.eps, momentum=0.2)
        )
        self.cnn_block = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=(3, 3), padding=1, stride=(1, 1),bias=False),
            GaussianActivationLayer(init_sigma=0.1 * random.random()),
            nn.BatchNorm2d(64, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            #
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=1),
            GaussianActivationLayer(init_sigma=0.1 * random.random()),
            nn.BatchNorm2d(32, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            #
            # nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            # GaussianActivationLayer(init_sigma=0.1 * random.random()),
            # nn.BatchNorm2d(128, eps=self.eps, momentum=0.2),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )


        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim_model),
            nn.LayerNorm(dim_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim_model, depth, heads, dim_head, mlp_dim, dropout)

        # self.to_latent = nn.Identity()

        self.mlp_head = MLPHead(dim_model, num_classes=num_classes)

    def forward(self, img):
        img = self.srm(img)
        img = self.cnn_block(img)
        # img = self.gauss(img)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = self.to_latent(x)

        output_class_token = x[:, 0]

        res = self.mlp_head(output_class_token)

        return res
