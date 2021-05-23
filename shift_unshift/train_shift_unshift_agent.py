import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import *
### path load

from importlib import import_module
import argparse

import numpy as np
import torch

from settings.shift_funcs import get_funcs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    # ウチはsettingを入れることにします
    parser.add_argument('--setting', default=None, type=str, help='setting file')
    parser.add_argument('--outdir', default=None, type=str, help='outdir')
    parser.add_argument('--data-file', default=None, type=str, help='dataset file')
    args = parser.parse_args()

    if args.setting is None:
        raise Exception('need setting')
    setting = import_module(f'settings.{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    outdir = setting.OUTDIR

    assert channel in [1, 2]
    
    if outdir is not None:
        outdir = os.path.join(os.path.abspath('results'), args.outdir)
    
    if args.data_file is None:
        raise Exception('require data dir.')
    data = np.load(os.path.join('data', args.data_file))
    train_dataset = data['train_dataset']
    train_dataset = torch.from_numpy(train_dataset).to(torch.float)
    original_dataset = data['original_dataset']
    original_dataset = torch.from_numpy(original_dataset).to(torch.float)

    train_func_labels = data['train_func_labels']
    funcs = [get_funcs(*delta) for delta in [(2, 2), (2, 0), (0, 2)]]
    actions = [lambda x: x] + [f[1] for f in funcs]

    for seed in range(args.start, args.end):
        train(train_dataset, original_dataset, actions, channel=channel, weight=weight, outdir=outdir, gpu=args.gpu, seed=seed)

