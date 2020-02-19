#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from os.path import join, isdir
from os import mkdir

import torch
from torch.utils.data import DataLoader
from PIL import Image
import tqdm

from utils import setup_devices, load_checkpoint
from dataset import create_default_splits
from model import Multi2Novel

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MultiView Training')
    parser.add_argument('--root', type=str, default="kitti",
                        help='dataset root to load')
    parser.add_argument('--out-dir', type=str, default="./sequence",
                        help='output image directort')
    parser.add_argument('--num-inputs', type=int, default=4,
                        help='number of aux views')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint to start from')
    parser.add_argument('--measure-time', action='store_true',
                        help='print forward pass time (awkwardly between progress bar lines)')
    args = parser.parse_args()

    device, _ = setup_devices(False, False, 13)

    _, dataset = create_default_splits(args.num_inputs, args.root, cache=False)

    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        pin_memory=True,
                        drop_last=False)

    model = Multi2Novel().to(device)

    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(model, None, filename=args.checkpoint.split(".")[0])

    if not isdir(args.out_dir):
        mkdir(args.out_dir)

    it = 0
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(loader), leave=False, desc="infer") as bar:
            for data in loader:
                (inp_img, inp_pose), (target_img, target_pose) = data
                inputs = (inp_img.to(device), inp_pose.to(device))
                targets = (target_img.to(device), target_pose.to(device))

                start = None
                end = None
                if args.measure_time:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                output = model(inputs, targets, train=False)[0]
                if args.measure_time:
                    end.record()
                    torch.cuda.synchronize()
                    print(start.elapsed_time(end))

                for aux in range(inputs[0].shape[1]):
                    img = ((inputs[0][0, aux].transpose(0,1).transpose(1,2) + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
                    pil = Image.fromarray(img.cpu().detach().numpy().astype('uint8'), 'RGB')
                    img_path = join(args.out_dir, f"frame_{it}_aux{aux}.png")
                    pil.save(img_path)

                img = ((output.transpose(0,1).transpose(1,2) + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
                pil = Image.fromarray(img.cpu().detach().numpy().astype('uint8'), 'RGB')
                img_path = join(args.out_dir, f"frame_{it}_pred.png")
                pil.save(img_path)

                img = ((targets[0][0].transpose(0,1).transpose(1,2) + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
                pil = Image.fromarray(img.cpu().detach().numpy().astype('uint8'), 'RGB')
                img_path = join(args.out_dir, f"frame_{it}_gt.png")
                pil.save(img_path)

                it += 1
                bar.update()
                bar.set_postfix(dict(total_it=it))

if __name__ == '__main__':
    main()
