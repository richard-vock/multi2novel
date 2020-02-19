#!/usr/bin/env python3

import argparse
import sys
import os
from os.path import join
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from numpy import inf
import shutil
import tqdm

from utils import setup_devices, load_checkpoint, save_checkpoint, checkpoint_state
from dataset import create_default_splits
from model import Multi2Novel
from loss import loss_fn

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MultiView Training')
    parser.add_argument('--root', type=str, default="kitti",
                        help='dataset root to load')
    parser.add_argument('--num-inputs', type=int, default=4,
                        help='number of aux views')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint to start from')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr-p', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-d', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-f', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--l2', type=float, default=5.0E-4)
    parser.add_argument('--decay-frequency', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true',
                        help='fixed seed PRNG')
    parser.add_argument('--seed', type=int, default=13, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--multi', action='store_true', default=False,
                        help='use all available GPUs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    device, dev_count = setup_devices(args.no_cuda, args.deterministic, args.seed)
    if not args.multi:
        dev_count = 1
    args.batch_size *= dev_count

    train_dataset, valid_dataset = create_default_splits(args.num_inputs, args.root, cache=args.multi)

    viz = SummaryWriter()

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.test_batch_size,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)

    model = Multi2Novel().to(device)

    flow_params = []
    pixel_params = []
    pixel_gan_params = []
    discr_params = []
    for k, v in model.named_parameters():
        if k.startswith('flow_predictor'):
            flow_params.append(v)
        elif k.startswith('pixel_generator'):
            pixel_params.append(v)
            pixel_gan_params.append(v)
        elif k.startswith('discriminator'):
            discr_params.append(v)
            pixel_gan_params.append(v)

    if args.multi:
        model = nn.DataParallel(model)

    opt_pixel = optim.Adam(pixel_params, lr=args.lr_p, weight_decay=args.l2)
    opt_pixel_gan = optim.Adam(pixel_gan_params, lr=args.lr_p, weight_decay=args.l2)
    opt_flow = optim.Adam(flow_params, lr=args.lr_f, weight_decay=args.l2)
    opt_discr = optim.Adam(discr_params, lr=args.lr_d, weight_decay=args.l2)

    it = 0
    epoch_start = 1
    best_loss = inf

    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(model, [opt_pixel, opt_pixel_gan, opt_flow, opt_discr], filename=args.checkpoint.split(".")[0])
        if checkpoint_status is not None:
            it, epoch_start, best_loss = checkpoint_status

    batch_count = len(train_loader)
    with tqdm.trange(epoch_start, args.epochs + 1, desc="epochs") as epoch_bar, tqdm.tqdm(
        total=batch_count, leave=False, desc="train"
    ) as train_bar:
        for epoch in epoch_bar:
            avg_train_loss = 0.0
            batch_it = 0.0
            for batch in train_loader:
                model.train()

                opt_pixel.zero_grad()
                opt_pixel_gan.zero_grad()
                opt_flow.zero_grad()
                opt_discr.zero_grad()

                (inp_img, inp_pose), (target_img, target_pose) = batch
                inputs = (inp_img.to(device), inp_pose.to(device))
                targets = (target_img.to(device), target_pose.to(device))
                outputs = model(inputs, targets)
                px_loss, fl_loss, px_gan_loss, d_loss = loss_fn(*outputs, targets[0])

                train_gan = it > 300000
                train_loss = 0.0
                opts = []
                if train_gan:
                    if it % 2 > 0: # train generator
                        train_loss = px_gan_loss + fl_loss
                        opts += [opt_pixel_gan, opt_flow]
                    else: # train discriminator
                        train_loss = d_loss
                        opts += [opt_discr,]
                else: # train pixel predictor + flow estimator
                    train_loss = px_loss + fl_loss
                    opts += [opt_pixel, opt_flow]

                if viz:
                    viz.add_scalar('loss/training', train_loss, it)

                train_loss.backward()
                for opt in opts:
                    opt.step()

                it += 1
                train_bar.update()
                train_bar.set_postfix(dict(total_it=it))
                epoch_bar.refresh()

                if (it % batch_count) == 0:
                    train_bar.close()

                    # validation
                    model.eval()
                    eval_dict = {}
                    val_loss = 0.0
                    for i, data in tqdm.tqdm(
                        enumerate(valid_loader, 0), total=len(valid_loader), leave=False, desc="val"
                    ):
                        loss = None
                        acc = None

                        with torch.no_grad():
                            (inp_img, inp_pose), (target_img, target_pose) = data
                            inputs = (inp_img.to(device), inp_pose.to(device))
                            targets = (target_img.to(device), target_pose.to(device))
                            outputs = model(inputs, targets)

                            px_loss, fl_loss, _, _ = loss_fn(*outputs, targets[0])
                            per_batch_loss = px_loss + fl_loss
                            val_loss += (per_batch_loss - val_loss) / (i+1)

                    if viz:
                        viz.add_scalar('loss/validation', val_loss, it)

                    is_best = val_loss < best_loss
                    best_loss = min(best_loss, val_loss)
                    save_checkpoint(
                        checkpoint_state(
                            model, [opt_pixel, opt_pixel_gan, opt_flow, opt_discr], val_loss, epoch, it, False
                        ),
                        is_best,
                        filename="checkpoints/step_ckpt",
                        bestname="checkpoints/best_ckpt",
                    )

                    train_bar = tqdm.tqdm(
                        total=batch_count, leave=False, desc="train"
                    )
                    train_bar.set_postfix(dict(total_it=it))

        if viz:
            viz.flush()
    #
    if viz is not None:
        viz.close()

if __name__ == '__main__':
    main()

