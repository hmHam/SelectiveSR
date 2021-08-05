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

# TODO:実験で指定された条件はこのオブジェクトとして管理するように修正する。
class Condition(object):
    '''実験条件のオブジェクト'''
    def __init__(self, args):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=4, type=str, help='setting file')
    parser.add_argument('--type', default=None, type=str, help='u d l r')
    parser.add_argument('--data-dir', default=None, type=str, help='data-dirが--typeと異なる時は指定する')
    parser.add_argument('--outdir', default=None, type=str, help='outdirが--typeと異なる時は指定する')
    parser.add_argument('--tri-num', default=20000, type=int, help='trial_num')
    # (FUNCS) _INVERT, _IRREV, _DIAG
    # (ACTIONS) _INVERT, _IRREV, _DISASTER
    parser.add_argument('--actions', required=True, type=str, help='行動')
    parser.add_argument('--funcs', required=True, type=str, help='変換')
    args = parser.parse_args()
    return args

def _load_train_data(args, train_path):
    # TODO: ルーチン化できそう
    train_dataset = np.load(train_path)
    train_Dy = train_dataset['train_dataset']
    train_Dy = torch.from_numpy(train_Dy).to(torch.float)
    train_Dx = train_dataset['original_dataset']
    train_Dx = torch.from_numpy(train_Dx).to(torch.float)

    # (assert) "訓練データ"で利用した変換と
    # ？訓練での変換 == 指定された変換(args.typeまたはargs.data_dirから算出)？
    train_labels = set(train_dataset['train_func_labels'])  # 訓練時に使用した関数
    types = {['u', 'd', 'l', 'r'].index(k) for k in re.sub('\d', '', (args.type if args.data_dir is None else args.data_dir).split('_')[0])}
    if train_labels != types:
        print('指定した train_dataset', train_path)
        print('type', args.type)
        raise ValueError('typeとデータセット対応してない')

    return train_Dy, train_Dx, types

def _load_test_data(args, test_path, types):
    # TODO: blur_unblurなどの他の実験でも使用できるようにルーチン化できそう。
    test_context = np.load(test_path)
    test_Dy = torch.from_numpy(test_context['test_dataset'])
    test_Dx = torch.from_numpy(test_context['original_dataset'])
    test_labels = set(test_context['test_func_labels'])

    # (assert) "テストデータ"版
    if types != test_labels:
        print('指定した test_dataset', test_path)
        print('type', args.type)
        # TODO: テストデータが対応していなかったため終了した旨を記録して終了するようにする
        raise ValueError('typeとデータセット対応してない')
    return test_Dy, test_Dx

def setup(args):
    # チャンネル数, 分類損失の重み
    if args.setting is None:
        raise Exception('you need setting.')
    setting = getattr(settings, f'setting{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    assert channel in [1, 2]

    # 行動集合
    funcs = getattr(shift_funcs, f'FUNCS_{args.funcs}')
    actions = [lambda x: x] + getattr(shift_funcs, f'ACTIONS_{args.actions}')

    # 訓練時の試行回数
    trial_num = args.tri_num

    # パスを指定(訓練, テストデータの読み込み先と学習済みモデル, 報酬, MSEの結果の出力先)
    data_dir = 'data/%s' % (args.type if args.data_dir is None else args.data_dir)
    outdir = os.path.abspath('results/%s' % (args.type if args.outdir is None else args.outdir))

    train_path = os.path.join(data_dir, 'train_dataset.npz')
    test_path = os.path.join(data_dir, 'test_dataset.npz')
    
    # 訓練データ
    train_Dy, train_Dx, types = _load_train_data(args, train_path)

    # 訓練データ
    test_Dy, test_Dx = _load_test_data(args, test_path, types)

    # (record) 実験条件を記録する。
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'condition.txt'), 'w') as f:
        # channel, weight, len(funcs), len(actions), trial_num, args.type, args.actions, args.funcs
        f.write(f'args.funcs: {args.funcs}\n')
        f.write(f'args.actions: {args.actions}\n')
        f.write(f'channel: {channel}\n')
        f.write(f'weight: {weight}\n')
        f.write(f'len(funcs): {len(funcs)}\n')
        f.write(f'len(action): {len(actions)}\n')
        f.write(f'trial_num: {trial_num}\n')
        f.write(f'train_dataset:, {train_path}\n')
        f.write(f'test_dataset:, {test_path}\n')

    return (
        train_Dy,
        train_Dx,
        test_Dy,
        test_Dx,
        actions,
        channel,
        weight,
        outdir,
        trial_num,
    )

def test(args, channel, weight, outdir, actions, trial_num, test_Dy, test_Dx):
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
        # metricsは、行方向にテストデータのサンプル[:, :-1]は選択した行動, [:, -1]はmse
        metrics = agent_metrics(test_Dy, test_Dx, Qnet, actions, channel=channel)
        mse = metrics[:, -1].mean()
        np.save(
            os.path.join(result_path, f'metrics{int(1000*mse):03d}'),
            metrics
        )
        print('metric end')


if __name__ == '__main__':
    args = parse_args()
    # cond = Condition(args)

    # 訓練
    (train_Dy, train_Dx, test_Dy, test_Dx, actions, channel, weight, outdir, trial_num) = setup(args)
    for seed in range(args.start, args.end):
        train(train_Dy, train_Dx, actions, channel, weight, outdir, trial_num, gpu=args.gpu, seed=seed)
    
    # (run) テストデータでエージェントの性能評価
    test(args, channel, weight, outdir, actions, trial_num, test_Dy, test_Dx)

