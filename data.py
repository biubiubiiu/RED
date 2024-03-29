import glob
import os.path as osp

import numpy as np
import rawpy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from utils import repeater


def train_dataloader(config):
    dataloader = DataLoader(
        SID(config, type='train'),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    return repeater(dataloader)


def val_dataloader(config):
    dataloader = DataLoader(
        SID(config, type='val'),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    return dataloader


def test_dataloader(config):
    dataloader = DataLoader(
        SID(config, type='test'),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    return dataloader


class SID(Dataset):

    def __init__(self, config, type='train'):
        super().__init__()

        self.config = config
        self.training = type == 'train'
        self.filter = getattr(config, f'{type}_data')

        lq_files = sorted(glob.glob(osp.join(config.data_dir, 'short', self.filter)))

        self.ids = list(map(lambda it: osp.basename(it)[:5], lq_files))  # match by image id
        gt_files = [glob.glob(osp.join(config.data_dir, 'long', f'{id}*.ARW'))[0]
                    for id in self.ids]

        assert len(lq_files) == len(gt_files)

        self.lq_rawarrays = []
        self.gt_imgs = []

        for lq_fname, gt_fname in tqdm(list(zip(lq_files, gt_files)),
                                       desc=f'Loading {type} data to RAM'):

            with rawpy.imread(lq_fname) as lq_raw:
                lq_arr = lq_raw.raw_image_visible.copy().astype(np.float32)
                lq_arr = np.maximum(lq_arr-512, 0) / (16383-512)  # subtract the black level
                amp = self.estimate_amplification(lq_fname, gt_fname, lq_arr)
                self.lq_rawarrays.append(lq_arr*amp)  # apply amplification

            with rawpy.imread(gt_fname) as gt_raw:
                gt_img = gt_raw.postprocess(use_camera_wb=True, half_size=False,
                                            no_auto_bright=True, output_bps=16)
                gt_img = (gt_img / 65535.).astype(np.float32)  # Rescale to [0, 1]
                gt_img = np.transpose(gt_img, [2, 0, 1])  # hwc -> chw
                self.gt_imgs.append(gt_img)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        lq = torch.from_numpy(self.lq_rawarrays[index]).unsqueeze(0)
        gt = torch.from_numpy(self.gt_imgs[index])
        h, w = lq.shape[1:]

        if self.training:
            ps = self.config.patch_size
            i, j, th, tw = RandomCrop.get_params(lq, (ps, ps))
            lq = T.crop(lq, i, j, th, tw)
            gt = T.crop(gt, i, j, th, tw)

            if torch.rand(1) < 0.5:  # 50% horizontal flip
                lq = T.hflip(lq)
                gt = T.hflip(gt)

            if torch.rand(1) < 0.2:  # 20% horizontal flip
                lq = T.vflip(lq)
                gt = T.vflip(gt)
        else:
            # The official implementation use cropping to make size multiple of 32.
            # Instead, we use reflection padding here.
            new_h, new_w = ((h + 32) // 32) * 32, ((w + 32) // 32) * 32
            pad_h = new_h - h if h % 32 != 0 else 0
            pad_w = new_w - w if w % 32 != 0 else 0
            lq = F.pad(lq.unsqueeze(0), (0, pad_w, 0, pad_h), 'reflect').squeeze(0)
            gt = F.pad(gt.unsqueeze(0), (0, pad_w, 0, pad_h), 'reflect').squeeze(0)

        return {'lq': lq, 'gt': gt, 'id': id, 'original_shape': (h, w)}

    def estimate_amplification(self, lq_fname, gt_fname, arr):
        if self.training or self.config.gt_amp:
            in_exposure = float(osp.basename(lq_fname)[9:-5])
            gt_exposure = float(osp.basename(gt_fname)[9:-5])
            coeff = min(gt_exposure / in_exposure, 300)
        else:
            coeff = get_na(arr)

        return coeff


def get_na(arr, amp=1.0):
    """Estimate the amplification factor"""

    bins = np.logspace(0, 8, 128, endpoint=True, base=2.0)  # create n bins
    bins = (bins-1)/255.

    weights = np.float32((np.logspace(0, 5, 127, endpoint=True, base=10.0)))
    weights = np.flip(weights)
    weights = weights / np.max(weights)

    selection_dict = {w: np.logical_and(bins[i] <= arr, arr < bins[i+1])
                      for i, w in enumerate(weights)}
    weights = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
    weights_sum = np.sum(weights, dtype=np.float64)

    del selection_dict

    na1 = np.float64(weights_sum*0.01*amp)/np.sum(arr*weights, dtype=np.float64)
    na1 = np.float32(na1)
    na1 = np.clip(na1, 1.0, 300.0)
    return na1
