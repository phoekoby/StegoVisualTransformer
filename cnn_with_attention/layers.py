import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class gaussian(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp):
        return torch.exp(-((inp - torch.mean(inp)) ** 2) / (torch.std(inp)) ** 2)


class SRMConv(nn.Module):
    def     __init__(self, device) -> None:
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


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_f, depth, kernel_size=3, padding=1, stride=1) -> None:
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_f,
            in_f * depth,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_f,
            stride=stride
        )

    def forward(self, x):
        out = self.depthwise(x)
        return out


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, depth_multiplier=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(
            in_channels,
            depth_multiplier,
            kernel_size=kernel_size,
            padding=padding
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier,
            out_channels,
            kernel_size=1,
            bias=bias,
            padding=0
        )
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.elu(out)
        return out


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=(1, 1), padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias)

        self.elu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.elu(out)
        return out


class GaussianActivationLayer(nn.Module):
    def __init__(self, init_sigma):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        out = torch.exp(torch.div(torch.multiply(torch.pow(x, 2), -0.5), self.sigma ** 2))
        return out
