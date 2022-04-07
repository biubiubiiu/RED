
import argparse
import os.path as osp
from datetime import timedelta

import torch
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image
from tqdm import tqdm

from data import train_dataloader, val_dataloader
from model import RED
from utils import Config, Timer, ensure_dir, init_env


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration')
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = Config(args.config)
    device = init_env(args, config)
    model = RED().to(device)
    optimizer = Adam(model.parameters(), lr=config.base_lr)
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    msssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)

    exp_dir = ensure_dir(config.exp_dir)

    train_loader, val_loader = train_dataloader(config), val_dataloader(config)
    timer = Timer()

    for iter in range(config.total_iters):
        with timer:
            model.train()
            data = next(train_loader)
            lq, gt = data['lq'].to(device), data['gt'].to(device)
            pred = model(lq)

            l1_loss = F.l1_loss(pred, gt)
            ssim_loss = 0.2*(1-msssim_loss(pred, gt))
            loss = l1_loss + ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if iter % config.log_step == 0:
            print(
                f'Iter [{iter}/{config.total_iters}]\t'
                f'eta: {timedelta(seconds=int(timer.average_time*(config.total_iters-iter-1)))}, '
                f'l1loss: {l1_loss.item():.3f}, '
                f'ssimloss: {ssim_loss.item():.3f}, '
                f'loss: {loss.item():.3f}')

        if iter % config.eval_step == 0:
            model.eval()
            with torch.no_grad():
                for data in tqdm(val_loader, desc='Evaluation', dynamic_ncols=True):
                    lq, gt, id = data['lq'].to(device), data['gt'].to(device), data['id'][0]
                    pred = model(lq)
                    save_path = ensure_dir(osp.join(config.exp_dir, 'val_result', f'{id:0>5}'))
                    save_image(pred, osp.join(save_path, f'iter_{iter}.png'))

        if iter % config.save_step == 0:
            torch.save(model.state_dict(), osp.join(exp_dir, f'iter_{iter}.pth'))

    torch.save(model.state_dict(), osp.join(exp_dir, 'final.pth'))


if __name__ == '__main__':
    main()
    