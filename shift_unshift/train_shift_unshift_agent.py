import sys
import os
import re
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import *
### path load

from importlib import import_module
import argparse

import numpy as np
import torch

from shift_funcs import FUNCS_INVERT, FUNCS_IRREV, ACTIONS_INVERT, ACTIONS_IRREV


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=None, type=str, help='setting file')
    parser.add_argument('--type', default=None, type=str, help='u d l r')
    parser.add_argument('--data-dir', default=None, type=str, help='data-dirが--typeと異なる時は指定する')
    parser.add_argument('--outdir', default=None, type=str, help='outdirが--typeと異なる時は指定する')
    args = parser.parse_args()
    
    # channel, weight
    if args.setting is None:
        raise Exception('you need setting.')
    setting = getattr(settings, f'{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    
    assert channel in [1, 2]
            
    ### data path & result path
    data_dir = 'data/%s' % (args.type if args.data_dir is None else args.data_dir)
    train_path = os.path.join(data_dir, 'train_dataset.npz')
    test_path = os.path.join(data_dir, 'test_dataset.npz')

    outdir = os.path.abspath('results/%s' % (args.type if args.outdir is None else args.outdir))

    ### 訓練データ
    train_dataset = np.load(train_path)
    Dy = train_dataset['train_dataset']
    Dy = torch.from_numpy(Dy).to(torch.float)
    Dx = train_dataset['original_dataset']
    Dx = torch.from_numpy(Dx).to(torch.float)
    # 訓練時に使用した関数とスクリプトで指定したtypeの整合性
    train_labels = set(train_dataset['train_func_labels'])  # 訓練時に使用した関数
    types = {['u', 'd', 'l', 'r'].index(k) for k in re.sub('\d', '', (args.type if args.data_dir is None else args.data_dir).split('_')[0])}
    if train_labels != types:
        print('指定した train_dataset', data_dir, 'train_dataset.npz')
        print('type', args.type)
        raise ValueError('typeとデータセット対応してない')
    ### Actionの候補
    if args.type.split('_')[-1] == 'invert':
        funcs = FUNCS_INVERT
        actions = [lambda x: x] + ACTIONS_INVERT
    else:
        funcs = FUNCS_IRREV
        actions = [lambda x: x] + ACTIONS_IRREV
    trial_num = 20000
    
    # テストデータ
    # device = 'cuda:%d' % args.gpu
    test_context = np.load(test_path)
    test_Dy = torch.from_numpy(test_context['test_dataset'])
    test_Dx = torch.from_numpy(test_context['original_dataset'])
    test_labels = set(test_context['test_func_labels'])
    if types != test_labels:
        print('指定した test_dataset', data_dir, 'test_dataset.npz')
        print('type', args.type)
        raise ValueError('typeとデータセット対応してない')

    ### 使用したtrain datasetとtest_datasetをtxtに記録する。
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'data_info.txt'), 'w') as f:
        f.write(f'train_dataset:, {train_path}\n')
        f.write(f'test_dataset:, {test_path}\n')
    ### train agent
    for seed in range(args.start, args.end):
        train(Dy, Dx, actions, channel=channel, weight=weight, trial_num=trial_num, outdir=outdir, gpu=args.gpu, seed=seed)
                
    # test 
    for seed in range(args.start, args.end):
        print('saving metrics...')
        print('seed', seed)
        result_path = os.path.join(
            outdir,
            'channel%02d_weight%03d_seed%02d' % (channel, int(100*weight), seed)
        )
        print('result_path', result_path)
        Qnet = QNet(c=channel, m=[20, 20, len(actions)]).to(device)
        Qnet.load_state_dict(torch.load(
            os.path.join(result_path, 'Qnet0%d.pth' % trial_num)
        ))
        metrics = agent_metrics(test_Dy, test_Dx, Qnet, actions, channel=channel)
        np.save(
            os.path.join(result_path, 'metrics.npz'),
            metrics
        )
        print('metric end')

