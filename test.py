
import argparse
import os.path as osp

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from data import test_dataloader
from metrics import compute_psnr, compute_ssim
from model import RED
from utils import AverageMeter, Timer, ensure_dir, init_env, parse_config


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration')
    parser.add_argument('checkpoint', type=str, help='Path to model weights')
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = parse_config(args.config)
    device = init_env(args, config)

    model = RED().to(device)

    ckp = torch.load(args.checkpoint)
    model.load_state_dict(ckp)

    psnr = AverageMeter('PSNR', ':6.3f')
    ssim = AverageMeter('SSIM', ':6.3f')
    batch_time = AverageMeter('Time', ':.4f')
    timer = Timer()

    exp_dir = ensure_dir(config.exp_dir)
    result_dir = ensure_dir(osp.join(exp_dir, 'results'))

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader(config), desc='Testing'):
            lq, gt = data['lq'].to(device), data['gt'].to(device)
            id, original_shape = data['id'][0], data['original_shape']

            if not args.cpu:
                torch.cuda.synchronize()

            with timer:
                pred = model(lq)
                pred = pred[:, :, :original_shape[0], :original_shape[1]]
                gt = gt[:, :, :original_shape[0], :original_shape[1]]

                if not args.cpu:
                    torch.cuda.synchronize()

            batch_time.update(timer.diff)
            psnr.update(compute_psnr(pred, gt, data_range=1.0))
            ssim.update(compute_ssim(pred, gt, data_range=1.0))

            save_image(pred, osp.join(result_dir, f'{id:0>5}.png'))

    for it in [psnr, ssim, batch_time]:
        print(it.summary())



if __name__ == '__main__':
    main()
    