import torch
import torch.nn as nn

from utils import Conv2d, ConvTranspose2d, ResNetLSTM
from encoder import Encoder

class PixelDecoder(nn.Module):
    def __init__(self, num_convs, lstm_scales):
        super().__init__()

        self.scales = lstm_scales

        dims_inp = [2 ** (5 + num_convs - i - 1) for i in range(num_convs)]
        self.dec = nn.ModuleList([
            ConvTranspose2d(2*d if i > 0 else d,
                            d // 2,
                            kernel_size=(3,3),
                            padding=(1,1),
                            output_padding=1,
                            stride=2,
                            bias=False,
                            bn=True)
            for i, d in enumerate(dims_inp)
        ])

        self.reduce = nn.Sequential(*[
            Conv2d(16, 8, kernel_size=(3,3), padding=(1,1), bias=False, bn=True),
            # 3+1 is 3 color channels + 1 confidence channel
            Conv2d( 8, 3+1, kernel_size=(3,3), padding=(1,1), bias=False, bn=False, activation=None)
        ])

    def forward(self, feats):
        x = None
        for i, conv in enumerate(self.dec):
            if i >= len(feats):
                x = conv(x)
            else:
                feat = feats[len(feats) - i - 1]
                if x is not None:
                    feat = torch.cat((feat, x), dim=1)
                x = conv(feat)

        x, confidence = self.reduce(x).split((3,1), dim=1)
        return torch.tanh(x), confidence

class PixelGenerator(nn.Module):
    def __init__(self, lstm_scales = 3, lstm_blocks = 2, num_convs = 5):
        super().__init__()

        self.lstm_scales = lstm_scales
        self.lstm_blocks = lstm_blocks

        self.enc = Encoder(num_convs)
        self.dec = PixelDecoder(num_convs, lstm_scales)

        feat_dims = [2 ** (5 + num_convs-i+1) for i in range(num_convs - lstm_scales, num_convs)]
        self.lstms = nn.ModuleList([ResNetLSTM(dim, k=3) for dim in feat_dims])

        self.state_list = []

    def forward(self, img, pose_delta):
        self.enc(img, pose_delta)

        for s in range(self.lstm_scales):
            layer = len(self.enc.features) - s - 1
            feat = self.enc.features[layer]

            for b in range(self.lstm_blocks):
                feat, self.state_list[s][b] = self.lstms[s](feat, self.state_list[s][b])

            self.enc.features[layer] = feat

        return self.dec(self.enc.features)

    def reset_state(self):
        self.state_list = []
        for s in range(self.lstm_scales):
            state_blocks = []
            for b in range(self.lstm_blocks):
                state_blocks.append((None, None))
            self.state_list.append(state_blocks)
