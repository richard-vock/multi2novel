import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_predictor import FlowPredictor
from pixel_generator import PixelGenerator
from discriminator import Discriminator

class Multi2Novel(nn.Module):
    def __init__(self, lstm_scales = 3, lstm_blocks = 2, num_convs = 5):
        super().__init__()

        self.flow_predictor = FlowPredictor(num_convs)
        self.pixel_generator = PixelGenerator(lstm_scales, lstm_blocks, num_convs)
        self.discriminator = Discriminator(4)

    def forward(self, inp, tgt, train=True):
        imgs, poses = inp
        target_img, target_pose = tgt
        n = poses.shape[1]
        pose_deltas = target_pose.unsqueeze(1).expand(-1, n, -1) - poses

        flow_imgs = []
        flow_confs = []
        for view in range(n):
            # flow prediction
            flow_img, flow_conf = self.flow_predictor(imgs[:,view], pose_deltas[:,view])
            flow_imgs.append(flow_img)
            flow_confs.append(F.softmax(flow_conf, dim=1))

            # pixel generation
            if view == 0:
                self.pixel_generator.reset_state()
            pixel_img, pixel_conf = self.pixel_generator(imgs[:, view], pose_deltas[:, view])
            pixel_conf = F.softmax(pixel_conf, dim=1)

        if train:
            combined_masks = torch.cat((pixel_conf, *flow_confs), dim=1).unsqueeze(2)
            combined_images = torch.cat((
                pixel_img.unsqueeze(1),
                *[img.unsqueeze(1) for img in flow_imgs]
            ), dim=1)
            aggregated = (combined_images * combined_masks).sum(dim=1)

            d_real = self.discriminator(target_img)[1]
            d_fake = self.discriminator(pixel_img)[1]
            return pixel_img, pixel_conf, flow_imgs, flow_confs, aggregated, d_real, d_fake
        else:
            h, w = pixel_conf.shape[2:]
            for i in range(len(flow_confs)):
                if h != flow_confs[i].shape[2] or w != flow_confs[i].shape[3]:
                    flow_confs[i] = F.interpolate(flow_confs[i], size=(h,w), mode='nearest')
                    flow_imgs[i] = F.interpolate(flow_imgs[i], size=(h,w), mode='nearest')
            all_masks = torch.cat((pixel_conf, *flow_confs), dim=1).unsqueeze(2)
            all_images = torch.cat((
                pixel_img.unsqueeze(1),
                *[img.unsqueeze(1) for img in flow_imgs]
            ), dim=1)
            return (all_images * all_masks).sum(dim=1)
