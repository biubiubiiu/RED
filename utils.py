import itertools
import os
import random
import time
from enum import Enum

import numpy as np
import toml
import torch
from addict import Dict
from torch.backends import cudnn


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
        self.diff = 0.

    @property
    def average_time(self):
        return sum(self.runtimes) / len(self.runtimes)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.diff = time.time() - self.start_time
        self.calls += 1
        self.total_time += self.diff
        self.runtimes.append(self.diff)
        if len(self.runtimes) > self.win_size:
            self.runtimes.pop(0)


def repeater(iterable):
    for it in itertools.repeat(iterable):
        for item in it:
            yield item


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def init_env(args, config):
    # set random seed
    if config.seed:
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


def parse_config(path):
    config = Dict(toml.load(path))
    return config
