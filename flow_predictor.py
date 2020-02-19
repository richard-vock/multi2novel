import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from utils import pixel_grid, Conv2d, ConvTranspose2d

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, k=3, stride=1):
        super().__init__()

        self.conv = Conv2d(in_planes, out_planes, kernel_size=(k,k), stride=stride, padding=(k//2,k//2), bias=False, bn=True)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.conv(inp)
        return self.activate(x - inp)

class FlowDecoder(nn.Module):
    def __init__(self, num_convs):
        super().__init__()

        self.dec = nn.Sequential(*[
            ConvTranspose2d(2 ** (5 + num_convs - i - 1), 2 ** (5 + num_convs - i - 2), kernel_size=(3,3), padding=(1,1), output_padding=1, stride=2, bias=False, bn=True) for i in range(num_convs)
        ])

        self.reduce = nn.Sequential(*[
            Conv2d(16, 8, kernel_size=(3,3), padding=(1,1), bias=False, bn=True),
            Conv2d( 8, 3, kernel_size=(3,3), padding=(1,1), bias=False, bn=False, activation=None)
        ])

    def forward(self, inp, imgs):
        x = self.dec(inp)
        x = self.reduce(x)

        dx, dy, mask = x.split(1, dim=1)

        # grid sample at px + (dx,dy)
        bs, _, h, w = dx.shape
        px_x, px_y = pixel_grid(bs, h, w, dx.device)
        px_x = (px_x + dx).transpose(1,2).transpose(2,3)
        px_y = (px_y + dy).transpose(1,2).transpose(2,3)
        grid = torch.cat((px_y, px_x), dim=3)
        pred = F.grid_sample(imgs, grid, align_corners=False)
        return pred, mask

class FlowPredictor(nn.Module):
    def __init__(self, num_convs, num_res_blocks = 4):
        super().__init__()

        # encoder
        self.enc = Encoder(num_convs)

        # residual subnet
        max_planes = 2 ** (5 + num_convs - 1)
        self.res = nn.Sequential(*[ResidualBlock(max_planes, max_planes) for i in range(num_res_blocks)])

        # decoder
        self.dec = FlowDecoder(num_convs)

    def forward(self, img, pose_delta):
        flow = self.enc(img, pose_delta)
        flow = self.res(flow)
        return self.dec(flow, img)
