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

import shift_funcs
# from shift_funcs import FUNCS_INVERT, FUNCS_IRREV, ACTIONS_INVERT, ACTIONS_IRREV


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=None, type=str, help='setting file')
    parser.add_argument('--type', default=None, type=str, help='u d l r')
    parser.add_argument('--data-dir', default=None, type=str, help='data-dirが--typeと異なる時は指定する')
    parser.add_argument('--outdir', default=None, type=str, help='outdirが--typeと異なる時は指定する')
    parser.add_argument('--tri-num', default=20000, type=int, help='trial_num')
    # (FUNCS) _INVERT, _IRREV, _DIAG
    # (ACTIONS) _INVERT, _IRREV, _DISASTER
    parser.add_argument('--actions', required=True, type=str, help='行動')
    parser.add_argument('--funcs', required=True, type=str, help='変換')
    args = parser.parse_args()
    
    ### エージェントの訓練に必要な情報をセットアップする。
    # (setup) チャンネル数, 分類損失の重み
    if args.setting is None:
        raise Exception('you need setting.')
    setting = getattr(settings, f'{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    
    assert channel in [1, 2]
            
    # (setup) パスを指定(訓練, テストデータの読み込み先と学習済みモデル, 報酬, MSEの結果の出力先)
    data_dir = 'data/%s' % (args.type if args.data_dir is None else args.data_dir)
    train_path = os.path.join(data_dir, 'train_dataset.npz')
    test_path = os.path.join(data_dir, 'test_dataset.npz')

    outdir = os.path.abspath('results/%s' % (args.type if args.outdir is None else args.outdir))

    # (setup) 訓練データの読み込み
    train_dataset = np.load(train_path)
    train_Dy = train_dataset['train_dataset']
    train_Dy = torch.from_numpy(train_Dy).to(torch.float)
    train_Dx = train_dataset['original_dataset']
    train_Dx = torch.from_numpy(train_Dx).to(torch.float)
    # (assert) "訓練データ"で利用した変換と
    # 実行時に想定している変換(args.typeなどから算出)が一致しているか確認
    train_labels = set(train_dataset['train_func_labels'])  # 訓練時に使用した関数
    types = {['u', 'd', 'l', 'r'].index(k) for k in re.sub('\d', '', (args.type if args.data_dir is None else args.data_dir).split('_')[0])}
    if train_labels != types:
        print('指定した train_dataset', data_dir, 'train_dataset.npz')
        print('type', args.type)
        raise ValueError('typeとデータセット対応してない')
    # (setup) エージェントの取る行動を選択
    # INVERT = 可逆シフト, IRREV = 不可逆シフト
    funcs = getattr(shift_funcs, f'FUNCS_{args.funcs}')
    actions = [lambda x: x] + getattr(shift_funcs, f'ACTIONS_{args.actions}')
    # (setup) 訓練時の試行回数
    trial_num = args.tri_num
    
    # (setup) テストデータの読み込み
    test_context = np.load(test_path)
    test_Dy = torch.from_numpy(test_context['test_dataset'])
    test_Dx = torch.from_numpy(test_context['original_dataset'])
    test_labels = set(test_context['test_func_labels'])
    # (assert) "テストデータ"で利用した変換と
    # 実行時に想定している変換(args.typeなどから算出)が一致しているか確認
    if types != test_labels:
        print('指定した test_dataset', data_dir, 'test_dataset.npz')
        print('type', args.type)
        raise ValueError('typeとデータセット対応してない')

    ### (record) どの訓練データとテストデータを使用したかを出力先に記録
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'data_info.txt'), 'w') as f:
        f.write(f'train_dataset:, {train_path}\n')
        f.write(f'test_dataset:, {test_path}\n')
    
    # (run) エージェントを訓練する
    for seed in range(args.start, args.end):
        train(train_Dy, train_Dx, actions, channel=channel, weight=weight, trial_num=trial_num, outdir=outdir, gpu=args.gpu, seed=seed)
                
    # (run) テストデータでエージェントの性能評価
    for seed in range(args.start, args.end):
        print('saving metrics...')
        print('seed', seed)
        result_path = os.path.join(
            outdir,
            'channel%02d_weight%03d_seed%02d' % (channel, int(100*weight), seed)
        )
        print('result_path', result_path)
        Qnet = QNet(c=channel, m=[20, 20, len(actions)])
        Qnet.load_state_dict(torch.load(
            os.path.join(result_path, 'Qnet0%d.pth' % trial_num)
        ))
        metrics = agent_metrics(test_Dy, test_Dx, Qnet, actions, channel=channel)
        mse = metrics[:, -1].mean()
        np.save(
            os.path.join(result_path, f'metrics{int(1000*mse):03d}'),
            metrics
        )
        print('metric end')

