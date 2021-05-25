# めんどい
# シフト1, setting1, setting2, setting3, setting4
# シフト2, setting1, setting2
# gpu 0

import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import *
### path load

from importlib import import_module
import numpy as np
import torch
from shift_funcs import get_funcs

shift_len = [2,]
# data_file = ['diag_dataset.npz', 'diag_hori_dataset.npz', 'diag_vert_dataset.npz']
data_file = ['diag_hori_dataset.npz', 'diag_vert_dataset.npz']

gpu = 1
device = 'cuda:%d' % (gpu,)
seed = 0

print("I'm working on", device)

for sl in shift_len:
    for dfname in data_file:
        for setting_i in [4]:
            print('--- shift%d' % sl, dfname, f'setting.{setting_i}')
            setting_fname = f'settings.setting{setting_i}'
            print('setting file', setting_fname)
            
            setting = import_module(setting_fname)
            channel = setting.CHANNEL
            weight = setting.WEIGHT
            outdir = setting.OUTDIR
            outdir = os.path.join('shift%d' % sl, dfname.split('dataset')[0].strip('_'))
            outdir = os.path.join(os.path.abspath('results'), outdir)
            print('outdir', outdir)

            data = np.load(os.path.join('data', 'shift%d' % sl, dfname))
            train_dataset = data['train_dataset']
            train_dataset = torch.from_numpy(train_dataset).to(torch.float)
            original_dataset = data['original_dataset']
            original_dataset = torch.from_numpy(original_dataset).to(torch.float)

            train_func_labels = data['train_func_labels']
            funcs = [get_funcs(*delta) for delta in [(sl, sl), (sl, 0), (0, sl)]]
            actions = [lambda x: x] + [f[1] for f in funcs]

            train(train_dataset, original_dataset, actions, channel=channel, weight=weight, outdir=outdir, gpu=gpu, seed=seed)
            
            # metrics and save
            test_context = np.load(
                os.path.join('data', 'shift%d' % sl, 'test_dataset.npz')
            )
            test_dataset = torch.from_numpy(test_context['test_dataset']).to(device)
            test_origin = torch.from_numpy(test_context['original_dataset']).to(device)
            
            result_path = os.path.join(
                outdir,
                'channel%02d_weight%03d_seed%02d' % (channel, int(100*weight), seed)
            )
            print('result_path', result_path)
            # 訓練済みモデルをロード
            Qnet = QNet(c=channel, m=[20, 20, len(actions)]).to(device)
            Qnet.load_state_dict(torch.load(
                 os.path.join(result_path, 'Qnet020000.pth')
            ))
            metrics = agent_metrics(test_dataset, test_origin, Qnet, actions, channel=channel, gpu=device)
            np.savez(
                os.path.join(result_path, 'metrics.npz'),
                took_actions=metrics[:, :-1],
                mse=metrics[:, -1]
            )
            print('metric end')