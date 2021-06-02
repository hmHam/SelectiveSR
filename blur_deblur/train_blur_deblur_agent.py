import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import *
### path load

import argparse

import numpy as np
import torch

from data import blur_funcs
# from blur_funcs import FUNCS_GR, ACTIONS_GR, FUNCS_RN, ACTIONS_RN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--actions-type', default='GR', type=str, help='GR, WO_RN_BLUR, RN')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=1, type=int, help='setting file')
    parser.add_argument('--data-file', default=None, type=str, help='train data_file/train_gauss_dataset.npz, train_with_random_dataset.npz, train_random_dataset.npz')
    parser.add_argument('--outdir', required=True, type=str, help='outdir')
    args = parser.parse_args()
    
    # channel, weight
    print('setting:', args.setting)
    setting = getattr(settings, f'setting{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    
    assert channel in [1, 2]
            
    ### 訓練データ
    if args.data_file is None:
        raise Exception('require data dir.')

    # outdir = os.path.join(args.actions_type, args.data_file.split('dataset')[0].strip('_'))
    outdir = os.path.join(os.path.abspath('results'), args.outdir)

    ### 訓練データ
    data = np.load(os.path.join('data', 'GR', args.data_file))
    train_dataset = data['train_dataset']
    train_dataset = torch.from_numpy(train_dataset).to(torch.float)
    original_dataset = data['original_dataset']
    original_dataset = torch.from_numpy(original_dataset).to(torch.float)

    ### Actionの候補
    funcs = getattr(blur_funcs, f'FUNCS_{args.actions_type}')
    actions = [lambda x: x] + getattr(blur_funcs, f'ACTIONS_{args.actions_type}')
    print(outdir)
    ### train
    for seed in range(args.start, args.end):
        train(train_dataset, original_dataset, actions, channel=channel, weight=weight, outdir=outdir, gpu=args.gpu, seed=seed)
