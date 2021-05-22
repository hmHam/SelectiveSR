import os
import argparse
import pickle
import numpy as np
from numpy.fft import fftshift, fft2
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal
from importlib import import_module

import torch
from torch import nn
from torch import optim

from settings.shift_funcs import get_funcs


def get_img(originals, deteriolated_imgs, channel=1):
    n = len(deteriolated_imgs)
    i = np.random.choice(n)
    y, x = originals[i], deteriolated_imgs[i]
    if channel == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif channel == 2:
        x = torch.cat([x, x], dim=0).unsqueeze(0)
    return (x, y.unsqueeze(0))


### reinforcement learning ###
# model #
class QNet(torch.nn.Module):
    def __init__(self, c=1, m=[20, 20, 5]):
        super(QNet, self).__init__()
        self.m = m
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c, self.m[0], 5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(m[0]),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(self.m[0], self.m[1], 5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(m[1]),
            torch.nn.MaxPool2d(2, stride=2)
        )
        self.OH = 4
        self.OW = self.OH
        self.fc = torch.nn.Linear(self.OH*self.OW*self.m[1], m[2])
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.OH*self.OW*self.m[1])
        x = self.fc(x)
        return x, self.scale*x
    
def next_state(s, a, filt, channel):
    x, y = s
    x_next = filt[a](x.numpy()[0, 0])
    x_next = torch.from_numpy(x_next).to(torch.float)
    if channel == 1:
        x_next = x_next.unsqueeze(0).unsqueeze(1)
    elif channel == 2:
        x_next = torch.stack([x_next, x[0, 1]], dim=0).unsqueeze(0)
    return (x_next, y)


def reward(s, s_next, a):
    x_next, y = s_next
    return - torch.mean((x_next[0, 0] - y[0, 0])**2).item()


def step(Qnet, s, history, filt, channel, lr=0.1, gamma=0.9, eps=0.1, device='cuda:0'):
    x, y = s
    
    # action -- eps greedy
    n = np.random.rand()
    if n < eps:
        a = np.random.randint(len(filt))
    else:
        with torch.no_grad():
            _, q = Qnet(x.to(device))
            q = q[0].to('cpu').detach().numpy()
        a = np.argmax(q)
    
    # reward
    with torch.no_grad():
        s_next = next_state(s, a, filt, channel)
        r_next = reward(s, s_next, a)
    
    history.append((s, s_next, a, r_next))
    return s_next, r_next, history


def subhistory(Qnet, history, idx):
    x, x_next, y, a, r = [], [], [], [], []
    for i in idx:
        s, s_next, b, r_next = history[i]
        x.append(s[0])
        x_next.append(s_next[0])
        y.append(s[1])
        a.append(b)
        r.append(r_next)
    x = torch.cat(x, dim=0)
    x_next = torch.cat(x_next, dim=0)
    y = torch.cat(y, dim=0)
    a = np.array(a)
    r = torch.tensor(r).to(torch.float)
    return x, x_next, y, a, r

    
### train ###
def train(deteriolated_data, original, actions, channel=1, weight=0.0, outdir='./', gpu=0, seed=0):
    device = 'cuda:%d' % (gpu,)

    # result directory & file
    dn = '%schannel%02d_weight%03d_seed%02d' % (outdir, channel, int(100*weight), seed)
    os.makedirs(dn, exist_ok=True)

    # setup
    torch.manual_seed(seed)
    Qnet = QNet(c=channel, m=[20, 20, len(actions)]).to(device)
    loss_fn = torch.nn.MSELoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Qnet.parameters(), lr=1e-3, momentum=0.9)

    print('preparation is done.')
    # parameters
    eps = 0.1
    gamma = 0.9
    batch = 50

    # training
    R = []
    history = []
    TRIAL_NUM = 20000
    FREQ = 500
    print('start trainning...')
    for itr in range(TRIAL_NUM):
        np.random.seed(itr)
        for _ in range(20):
            Ri = 0
            s = get_img(original, deteriolated_data, channel)
            for i in range(5):
                s, r, history = step(Qnet, s, history, actions, channel, gamma=gamma, eps=eps, device=device)
                Ri = Ri + r
            R.append(Ri)

        if len(history) < batch:
            continue

        with torch.no_grad():
            idx = np.random.choice(len(history), batch)
            x, x_next, y, a, r = subhistory(Qnet, history, idx)

            # loss - Q
            _, q = Qnet(x_next.to(device))
            q = r.to(device) + gamma * torch.max(q, dim=1)[0]

            # loss - clf
            t = []
            for xn, yn in zip(x, y):
                xn, yn = xn[0].numpy(), yn.numpy()
                X = np.stack([f(xn) for f in actions], axis=0)
                err = np.sum((X - yn)**2, axis=(1, 2))
                b = np.argmin(err)
                t.append(b)
            t = torch.tensor(t)

        x, q, t = x.to(device), q.to(device), t.to(device)
        z, p = Qnet(x)
        loss1 = 0
        for b in range(len(actions)):
            idx = np.where(a == b)[0]
            if idx.size == 0:
                continue
            loss1 = loss1 + loss_fn(p[idx, b], q[idx])
        loss2 = loss_fn2(z, t)
        loss = loss1 + weight * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save
        if (itr+1) % FREQ == 0:
            torch.save(Qnet.state_dict(), '%s/Qnet%06d.pth' % (dn, itr+1))
            print('iteration: ', itr+1)
            with open('%s/reward.pkl' % (dn,), 'wb') as f:
                pickle.dump(R, f)
    print('trainning is done.')

    
### main ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
#     parser.add_argument('--channel', default=1, type=int, help='number of channels')
#     parser.add_argument('--weight', default=0, type=float, help='weight of loss2')
#     parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=1, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    parser.add_argument('--setting', default=None, type=str, help='setting file')
    args = parser.parse_args()
    
    if args.setting is None:
        raise Exception('need setting')
    setting = import_module(f'settings.{args.setting}')
    channel = setting.CHANNEL
    weight = setting.WEIGHT
    outdir = setting.OUTDIR
    
    assert channel in [1, 2]
    data = np.load('data/shift_dataset.npz')
    train_dataset = data['train_dataset']
    train_dataset = torch.from_numpy(train_dataset).to(torch.float)
    original_dataset = data['original_dataset']
    original_dataset = torch.from_numpy(original_dataset).to(torch.float)
    
    train_func_labels = data['train_func_labels']
    funcs = [get_funcs(*delta) for delta in [(2, 2), (2, 0), (0, 2)]]
    actions = [lambda x: x] + [f[1] for f in funcs]
    for seed in range(args.start, args.end):
        train(train_dataset, original_dataset, actions, channel=channel, weight=weight, outdir=outdir, gpu=args.gpu, seed=seed)
        
