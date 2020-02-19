import torch
import torch.nn as nn
from utils import Conv2d

class Discriminator(nn.Module):
    def __init__(self, num_convs):
        super().__init__()

        self.enc = nn.Sequential(*[
            Conv2d(2**(4+i) if i > 0 else 3,
                   2**(5+i),
                   kernel_size=(3,3),
                   padding=(1,1),
                   stride=2,
                   activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                   bias=False,
                   bn=True) for i in range(num_convs)
        ])

        self.pred = Conv2d(2**(5+num_convs-1),
                           1,
                           kernel_size=(1,1),
                           padding=(0,0),
                           stride=1,
                           activation=None,
                           bias=False,
                           bn=True)

    def forward(self, inp):
        x = self.enc(inp)
        x = self.pred(x)
        return torch.sigmoid(x), x
