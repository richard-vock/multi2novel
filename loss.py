import torch
import torch.nn.functional as F

def confidence_loss(img, conf, target_img):
    # this computes a L1 loss as in the original TF implementation.
    # note that the paper states that a L2 loss is used
    bs, _, h, w = img.shape

    normalized_mask = F.normalize(conf.view(bs, -1)).view(bs, h, w)
    loss_map = F.l1_loss(img, target_img, reduction='none').mean(dim=1)
    return (loss_map * normalized_mask).mean() * 0.01 / (w*h)

def loss_fn(pixel_img, pixel_conf, flow_imgs, flow_confs, aggr_img, d_real, d_fake, target_img):
    # adversarial loss
    L_r = F.mse_loss(d_real, torch.ones_like(d_real))
    L_d = L_r + F.mse_loss(d_fake, torch.zeros_like(d_fake))
    L_g = F.mse_loss(d_fake, torch.ones_like(d_fake))

    # image loss
    L_p = F.l1_loss(pixel_img, target_img)
    L_f = sum([F.l1_loss(img, target_img) for img in flow_imgs]) / len(flow_imgs)
    L_a = F.l1_loss(aggr_img, target_img)

    # confidence weighted loss
    L_c_pixel = confidence_loss(pixel_img, pixel_conf, target_img)
    L_c_flow = sum([confidence_loss(img, mask, target_img)
               for img, mask in zip(flow_imgs, flow_confs)]) / len(flow_imgs)

    # factor 10 is cargo-culted from the original TF implementation
    pixel_loss = 10 * (L_p + L_a + L_c_pixel)
    flow_loss = L_f + L_a + L_c_flow

    return pixel_loss, flow_loss, pixel_loss + L_g, L_d
