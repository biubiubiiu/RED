import itertools
import os
import random
import time
from enum import Enum

import numpy as np
import toml
import torch
from torch.backends import cudnn


class Config:
    def __init__(self, path: str):
        super(Config, self).__init__()

        raw_config = toml.load(path)
        self.attr = _SimpleBunch(raw_config)

    def __getattr__(self, item):
        if hasattr(self.attr, item):
            return self.attr.__getattribute__(item)
        else:
            return None

    def __repr__(self):
        return self.attr.__repr__()


class _SimpleBunch:
    """Recursively transforms a dictionary into a Bunch via copy."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [_SimpleBunch(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, _SimpleBunch(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return '<' + str('\n '.join(f'{k} : {repr(v)}' for (k, v) in self.__dict__.items())) + '>'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is self.Summary.NONE:
            fmtstr = ''
        elif self.summary_type is self.Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is self.Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is self.Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError(f'invalid summary type {self.summary_type}')

        return fmtstr.format(**self.__dict__)


class Timer(object):
    """A simple timer."""

    def __init__(self, win_size=200):
        assert win_size >= 1
        self.win_size = win_size
        self.runtimes = []
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.

    @property
    def average_time(self):
        return sum(self.runtimes) / len(self.runtimes)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        diff = time.time() - self.start_time
        self.calls += 1
        self.total_time += diff
        self.runtimes.append(diff)
        if len(self.runtimes) > self.win_size:
            self.runtimes.pop(0)


def repeater(data_loader):
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def init_env(args, config):
    # set random seed
    if config.seed is not None:
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    if config.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False

    # setup environment
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    return device


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
