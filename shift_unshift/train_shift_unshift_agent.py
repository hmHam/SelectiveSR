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

def train_shift_unshift():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=None, type=str, help='setting file')
    parser.add_argument('--shift-len', default=None, type=int, help='shift length')
    parser.add_argument('--data-file', default=None, type=str, help='dataset file')
    args = parser.parse_args()
    
    # channel, weight
    if args.setting is None:
        raise Exception('you need setting.')
    setting = import_module(f'settings.{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    
    assert channel in [1, 2]
            
    ### 出力先
    if args.shift_len is None:
        raise Exception('you need shift length.')     
    SL = args.shift_len
    ### 訓練データ
    if args.data_file is None:
        raise Exception('require data dir.')

    outdir = os.path.join('shift%d' % SL, args.data_file.split('dataset')[0].strip('_'))
    outdir = os.path.join(os.path.abspath('results'), outdir)

    ### 訓練データ
    data = np.load(os.path.join('data', 'shift%d' % SL, args.data_file))
    train_dataset = data['train_dataset']
    train_dataset = torch.from_numpy(train_dataset).to(torch.float)
    original_dataset = data['original_dataset']
    original_dataset = torch.from_numpy(original_dataset).to(torch.float)
#     train_func_labels = data['train_func_labels'] # 使わない

    ### Actionの候補
    funcs = [get_funcs(*delta) for delta in [
        (SL, SL),
        (SL, -SL),
        (-SL, SL),
        (-SL, -SL),
    ]]
    actions = [lambda x: x] + [f[1] for f in funcs]
    
    ### train
    for seed in range(args.start, args.end):
        train(train_dataset, original_dataset, actions, channel=channel, weight=weight, outdir=outdir, gpu=args.gpu, seed=seed)
        
                
#     # metrics and save
#     test_context = np.load(
#         os.path.join('data', 'shift%d' % SL, 'test_dataset.npz')
#     )
#     test_dataset = torch.from_numpy(test_context['test_dataset']).to(device)
#     test_origin = torch.from_numpy(test_context['original_dataset']).to(device)

#     result_path = os.path.join(
#         outdir,
#         'channel%02d_weight%03d_seed%02d' % (channel, int(100*weight), seed)
#     )
#     print('result_path', result_path)
#     # 訓練済みモデルをロード
#     Qnet = QNet(c=channel, m=[20, 20, len(actions)]).to(device)
#     Qnet.load_state_dict(torch.load(
#          os.path.join(result_path, 'Qnet020000.pth')
#     ))
#     metrics = agent_metrics(test_dataset, test_origin, Qnet, actions, channel=channel, gpu=device)
#     np.savez(
#         os.path.join(result_path, 'metrics.npz'),
#         took_actions=metrics[:, :-1],
#         mse=metrics[:, -1]
#     )
#     print('metric end')

