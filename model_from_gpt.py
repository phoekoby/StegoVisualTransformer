import torch
from torch import nn

from model import SRMConv


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=30, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.GELU(),
            nn.Linear(emb_size*4, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_size=768, n_classes=2, depth=12, device=None):
        super().__init__()
        self.srm = SRMConv(device=device)
        self.patch_emb = PatchEmbedding(img_size=img_size, patch_size=patch_size, emb_size=emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + (img_size//patch_size)**2, emb_size))
        self.encoders = nn.Sequential(*[TransformerEncoder(emb_size=emb_size) for _ in range(depth)])
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.srm(x)
        n_samples = x.shape[0]
        x = self.patch_emb(x)
        cls_tokens = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb
        x = self.encoders(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)
