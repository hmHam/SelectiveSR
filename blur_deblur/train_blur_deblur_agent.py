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


def augument_data(Dy, Dx, seed=0):
    np.random.seed(seed)
    N = Dy.shape[0]
    thetaN = np.random.randint(1, 4, N)
    Dy_augumented = []
    Dx_augumented = []
    for n in range(Dy.shape[0]):
        yn = Dy[n]
        theta = thetaN[n]

        Dy_new = [
            yn,
            np.rot90(yn, theta),
            np.fliplr(yn),
            np.flipud(yn)                
        ]
        Dy_augumented.extend(Dy_new)

        xn = Dx[n]
        Dx_new = [
            xn,
            np.rot90(xn, theta),
            np.fliplr(xn),
            np.flipud(xn)
        ]
        Dx_augumented.extend(Dx_new)
    Dy_augumented = np.array(Dy_augumented, dtype=Dy.dtype)
    Dx_augumented = np.array(Dx_augumented, dtype=Dx.dtype)
    return Dy_augumented, Dx_augumented

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--actions-type', default='GR', type=str, help='GR, WO_RN_BLUR, RN')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=1, type=int, help='setting file')
    parser.add_argument('--data-file', default=None, type=str, help='train data_file/train_gauss_dataset.npz, train_with_random_dataset.npz, train_random_dataset.npz')
    parser.add_argument('--outdir', required=True, type=str, help='outdir')
    parser.add_argument('--augument', type=bool, default=False)
    parser.add_argument('--trial-num', type=int, default=20000)
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
    data = np.load(os.path.join('data', args.data_file))
    Dy = data['train_dataset']
    Dx = data['original_dataset']
    if args.augument:
        Dy, Dx = augument_data(Dy, Dx)
    print('Data size after augmentation = ', Dy.shape[0])

    Dy = torch.from_numpy(Dy).to(torch.float)
    Dx = torch.from_numpy(Dx).to(torch.float)

    ### Actionの候補
    # funcs = getattr(blur_funcs, f'FUNCS_{args.actions_type}')
    actions = [lambda x: x] + getattr(blur_funcs, f'ACTIONS_{args.actions_type}')
    print(len(actions))
    print(outdir)
    ### train
    for seed in range(args.start, args.end):
        train(Dy, Dx, actions, channel=channel, weight=weight, outdir=outdir, trial_num=args.trial_num, gpu=args.gpu, seed=seed)
