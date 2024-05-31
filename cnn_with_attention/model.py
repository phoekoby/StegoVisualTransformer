import random

import torch
from torch import nn

from attention import ProjectorBlock, SpatialAttn
from layers import ConvBn, GaussianActivationLayer, SRMConv


class SRMKovNet(nn.Module):

    def __init__(self, attention=True, sample_size=32, device=None) -> None:
        super(SRMKovNet, self).__init__()
        self.eps = 0.001
        self.attention = attention

        self.srm = SRMConv(device=device)
        self.batchN1 = nn.BatchNorm2d(30, eps=self.eps, momentum=0.2)
        self.first_block = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=(3, 3), padding=1, stride=(1, 1),
                      bias=False),
            GaussianActivationLayer(init_sigma=0.1 * random.random()),
            nn.BatchNorm2d(64, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

        self.second_block = nn.Sequential(
            ConvBn(64, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )
        self.third_block = nn.Sequential(
            ConvBn(128, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256, eps=self.eps, momentum=0.2),
        )
        self.fourth_block = nn.Sequential(
            ConvBn(256, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=int(32), padding=0, bias=True),
            nn.Dropout(p=0.5)
        )
        if self.attention is True:
            self.projector_1 = ProjectorBlock(64, 256)
            self.projector_2 = ProjectorBlock(128, 256)
            self.attn1 = SpatialAttn(in_features=256, normalize_attn=True)
            self.attn2 = SpatialAttn(in_features=256, normalize_attn=True)
            self.attn3 = SpatialAttn(in_features=256, normalize_attn=True)
        if self.attention:
            self.classify = nn.Linear(in_features=256 * 3, out_features=2, bias=True)
        else:
            self.classify = nn.Linear(in_features=256, out_features=2, bias=True)

    def forward(self, inp):
        out_srm = self.srm(inp)
        out = self.batchN1(out_srm)
        out_first_block = self.first_block(out)
        out_second_block = self.second_block(out_first_block)
        out_third_block = self.third_block(out_second_block)
        out = self.fourth_block(out_third_block)
        if self.attention is True:
            c1, g1 = self.attn1(self.projector_1(out_first_block), out)
            c2, g2 = self.attn2(self.projector_2(out_second_block), out)
            c3, g3 = self.attn3(out_third_block, out)
            out = torch.cat((g1, g2, g3), dim=1)
            out = self.classify(out)
        else:
            c_crm, c1, c2, c3 = None, None, None, None
            out = self.classify(torch.squeeze(out))
        return [out, c1, c2, c3]


class SRMKovTransformerCNN(nn.Module):
    def __init__(self, attention=True, sample_size=32, device=None) -> None:
        super(SRMKovNet, self).__init__()
        self.eps = 0.001
        self.attention = attention

        self.srm = SRMConv(device=device)
        self.batchN1 = nn.BatchNorm2d(30, eps=self.eps, momentum=0.2)
        self.first_block = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=(3, 3), padding=1, stride=(1, 1),
                      bias=False),
            GaussianActivationLayer(init_sigma=0.1 * random.random()),
            nn.BatchNorm2d(64, eps=self.eps, momentum=0.2),

            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

        self.second_block = nn.Sequential(
            ConvBn(64, 128, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128, eps=self.eps, momentum=0.2),
            nn.MultiheadAttention(embed_dim=128, num_heads=32),

            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )
        self.third_block = nn.Sequential(
            ConvBn(128, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256, eps=self.eps, momentum=0.2),
            nn.MultiheadAttention()
        )
        self.fourth_block = nn.Sequential(
            ConvBn(256, 256, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256, eps=self.eps, momentum=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=int(32), padding=0, bias=True),
            nn.Dropout(p=0.5)
        )
        if self.attention is True:
            self.projector_1 = ProjectorBlock(64, 256)
            self.projector_2 = ProjectorBlock(128, 256)
            self.attn1 = SpatialAttn(in_features=256, normalize_attn=True)
            self.attn2 = SpatialAttn(in_features=256, normalize_attn=True)
            self.attn3 = SpatialAttn(in_features=256, normalize_attn=True)
        if self.attention:
            self.classify = nn.Linear(in_features=256 * 3, out_features=2, bias=True)
        else:
            self.classify = nn.Linear(in_features=256, out_features=2, bias=True)

    def forward(self, inp):
        out_srm = self.srm(inp)

        out = self.batchN1(out_srm)

        out_first_block = self.first_block(out)
        out_second_block = self.second_block(out_first_block)

        out_third_block = self.third_block(out_second_block)

        out = self.fourth_block(out_third_block)
        if self.attention is True:
            c1, g1 = self.attn1(self.projector_1(out_first_block), out)
            c2, g2 = self.attn2(self.projector_2(out_second_block), out)
            c3, g3 = self.attn3(out_third_block, out)
            out = torch.cat((g1, g2, g3), dim=1)
            out = self.classify(out)
        else:
            c_crm, c1, c2, c3 = None, None, None, None
            out = self.classify(torch.squeeze(out))
        return [out, c1, c2, c3]
