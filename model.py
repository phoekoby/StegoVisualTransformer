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
            dim_feedforward=32,
            blocks=2,
            mlp_dim=1024,
            n_classes=2,
            dropout=0.5,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = 30
        # self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.blocks = blocks
        self.mlp_dim = mlp_dim
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.srm = SRMConv(device=device)

        self.patch_embedding = PatchEmbedding(
            image_size=img_size,
            patch_size=self.patch_size,
            in_channels=self.n_channels,
            hidden_dim=self.hidden_dim,
            dropout=0.5,
            device=device
        )

        # self.img2seq = Img2Seq(self.img_size, self.patch_size, self.n_channels, self.hidden_dim, device=device)
        self.encoder = Encoder(
            seq_length=(self.img_size // self.patch_size) ** 2 + 1,
            num_layers=self.blocks,
            num_heads=self.nhead,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout=0.5,
            attention_dropout=0.5
        )
        # encoder_layer = nn.TransformerEncoderLayer(
        #     self.hidden_dim, self.nhead, self.dim_feedforward, activation="elu", batch_first=True
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer, self.blocks
        # )
        #
        # self.mlp = get_mlp(self.hidden_dim,  self.mlp_head_units, self.n_classes)

        self.output = nn.Softmax(dim=-1)

    def forward(self, inp):
        out = self.srm(inp)
        out = self.patch_embedding(out)

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
