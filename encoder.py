import torch
import torch.nn as nn
from utils import pixel_grid, Conv2d

class Encoder(nn.Module):
    def __init__(self, num_convs):
        super().__init__()

        self.enc = nn.ModuleList([
            Conv2d(2 ** (i + 4) if i > 0 else 11, 2 ** (i + 5), kernel_size=(3,3), padding=(1,1), stride=2, bias=False, bn=True) for i in range(num_convs)
        ])

        self.features = []

    def forward(self, images, pose_deltas):
        bs, pose_dim = pose_deltas.shape
        height, width = images.shape[2:]

        px_x, px_y = pixel_grid(bs, height, width, images.device)

        pose_deltas = pose_deltas.view(bs, pose_dim, 1, 1).expand(-1, -1, height, width)

        feat = torch.cat((images, px_x, px_y, pose_deltas), dim=1)

        self.features = []
        for conv in self.enc:
            feat = conv(feat)
            self.features.append(feat)

        return feat

