import os
import argparse
import pickle
import numpy as np
from numpy.fft import fftshift, fft2
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal
from importlib import import_module
from collections import deque

import torch
from torch import nn
from torch import optim
from .wrap_func import dilig


def get_img(Dx, Dy, channel=1):
    n = len(Dy)
    i = np.random.choice(n)
    x, y = Dx[i], Dy[i]
    if channel == 1:
        s = y.unsqueeze(0).unsqueeze(0)
    elif channel == 2:
        s = torch.stack([y, y], dim=0).unsqueeze(0)
    return (s, x)


# model #
# class QNet(torch.nn.Module):
#     def __init__(self, c=1, m=[20, 20, 5]):
#         super(QNet, self).__init__()
#         self.m = m
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(c, self.m[0], 5, stride=1, padding=0),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm2d(m[0]),
#             torch.nn.MaxPool2d(2, stride=2),
#             torch.nn.Conv2d(self.m[0], self.m[1], 5, stride=1, padding=0),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm2d(m[1]),
#             torch.nn.MaxPool2d(2, stride=2)
#         )
#         self.OH = 4
#         self.OW = self.OH
#         self.fc = torch.nn.Linear(self.OH*self.OW*self.m[1], m[2])
#         self.scale = torch.nn.Parameter(torch.tensor(1.0))

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, self.OH*self.OW*self.m[1])
#         x = self.fc(x)
#         return x, self.scale*x


class QNet(torch.nn.Module):
    def __init__(self, c=1, m=[20, 20, 5]):
        super(QNet, self).__init__()
        self.m = m
        self.conv1 = torch.nn.Conv2d(c, self.m[0], 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(self.m[0], self.m[1], 5, stride=1, padding=0)
        self.fc = torch.nn.Linear(4*4*self.m[1], m[2])
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.m[1])
        x = self.fc(x)
        return x, self.scale*x
    
def next_state(s_x, a, actions, channel):
    s, x = s_x
    z, y = s[0]
    z_next = actions[a](z.numpy())
    z_next = torch.from_numpy(z_next).to(torch.float)
    if channel == 1:
        s_next = z_next.unsqueeze(0).unsqueeze(0)
    elif channel == 2:
        s_next = torch.stack([z_next, y], dim=0).unsqueeze(0)
    return (s_next, x)


def reward(s_x, s_next_x, a):
    s_next, x = s_next_x
    z_next = s_next[0, 0]
    return - torch.mean((z_next - x)**2).item()


# def reward(s_x, s_next_x, a):
#     s_next, x = s_next_x
#     z_next = s_next[0, 0]
#     return torch.mean((z_next * x)).item()


# NOTE: 元の報酬関数を、ステップ数が後半であるほど高い値を与えるよう重みづけした。
# WARNING: ↑これは全く必要ない工夫。なぜならAgentのアウトプットのMSEが小さければ
#          途中で何を選んだかなんて何も関係ない。
#          想定通りの行動を選択して欲しいのは、実験をする側の都合
#          想定通りに動作するように実験の問題の方を修正する必要がある。
# def get_dxdy(sl, j):
#     return [(0, sl), (0, -sl), (-sl, 0), (sl, 0)][j]

# ### シフト幅reward
# def reward(s_x, s_next_x, a):
#     s_next, x = s_next_x
#     z_next = s_next[0, 0]
#     min_sl = 0
#     min_mse = torch.mean((z_next - x)**2).item()
#     for i in range(1, 15):  # シフト数
#         for j in range(4):  # 上下左右
#             o = z_next.clone()
#             dx, dy = get_dxdy(i, j)
#             o = dilig(lambda A: np.roll(A, (dy, dx), axis=(0, 1)))(o)
#             mse = torch.mean((o - x)**2).item()
#             if mse < min_mse:
#                 min_mse = mse
#                 min_sl = i
#     return - min_sl

def step(Qnet, s_x, history, actions, channel, lr=0.1, gamma=0.9, eps=0.1, device='cuda:0'):
    s, x = s_x
    
    # action -- eps greedy
    n = np.random.rand()
    if n < eps:
        a = np.random.randint(len(actions))
    else:
        with torch.no_grad():
            _, q = Qnet(s.to(device))
            q = q[0].to('cpu').detach().numpy()
        a = np.argmax(q)
    
    # reward
    with torch.no_grad():
        s_next_x = next_state(s_x, a, actions, channel)
        r = reward(s_x, s_next_x, a)
    
    history.append((s_x, s_next_x, a, r))
    return s_next_x, r, history


def subhistory(Qnet, history, idx):
    sN, s_nextN, xN, aN, rN = [], [], [], [], []
    for i in idx:
        s_x, s_next_x, b, r = history[i]
        s, s_next = s_x[0], s_next_x[0]
        x = s_x[1]
        sN.append(s)
        s_nextN.append(s_next)
        xN.append(x)
        aN.append(b)
        rN.append(r)
    sN = torch.cat(sN, dim=0)
    s_nextN = torch.cat(s_nextN, dim=0)
    xN = torch.stack(xN, dim=0)
    aN = np.array(aN)
    rN = torch.tensor(rN).to(torch.float)
    return sN, s_nextN, xN, aN, rN

    
### train ###
def train(Dy, Dx, actions, channel=1, weight=0.0, outdir='./', trial_num=20000, gpu=0, seed=0):
    assert os.path.isabs(outdir), 'give me absolute path as outdir'
    device = 'cuda:%d' % (gpu,)

    # result directory & file
    dn = os.path.join(outdir, 'channel%02d_weight%03d_seed%02d' % (channel, int(100*weight), seed))
    os.makedirs(dn, exist_ok=True)

    # setup
    torch.manual_seed(seed)
    Qnet = QNet(c=channel, m=[20, 20, len(actions)]).to(device)
    loss_fn = torch.nn.MSELoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Qnet.parameters(), lr=1e-3, momentum=0.9)

    # parameters
    eps = 0.1
    gamma = 0.9
    batch = 50
    # T = 16
    T = 5

    # training
    R = []
    # history = deque(maxlen=1024)
    history = []
    print(type(history))
    TRIAL_NUM = trial_num
    FREQ = 500
    print('start trainning...')
    for itr in range(TRIAL_NUM):
        np.random.seed(itr)
        for _ in range(20):
            Ri = 0
            s_x = get_img(Dx, Dy, channel)
            for i in range(T):
                s_x, r, history = step(Qnet, s_x, history, actions, channel, gamma=gamma, eps=eps, device=device)
                Ri = Ri + r
            R.append(Ri)

        if len(history) < batch:
            continue

        # update parameter
        with torch.no_grad():
            idx = np.random.choice(len(history), batch)
            sN, s_nextN, xN, aN, rN = subhistory(Qnet, history, idx)

            # loss - Q
            _, q_nextN = Qnet(s_nextN.to(device))
            q_nextN = rN.to(device) + gamma * torch.max(q_nextN, dim=1)[0]

            # loss - clf
            tN = []
            for sn, xn in zip(sN, xN):
                zn, xn = sn[0].numpy(), xn.numpy()
                Z = np.stack([f(zn) for f in actions], axis=0)
                err = np.sum((Z - xn)**2, axis=(1, 2))
                b = np.argmin(err)
                tN.append(b)
            tN = torch.tensor(tN)

        sN, q_nextN, tN = sN.to(device), q_nextN.to(device), tN.to(device)
        pN, qN = Qnet(sN)
        loss1 = 0
        for b in range(len(actions)):
            idx = np.where(aN == b)[0]
            if idx.size == 0:
                continue
            loss1 = loss1 + loss_fn(qN[idx, b], q_nextN[idx])
        loss2 = loss_fn2(pN, tN)
        loss = loss1 + weight * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save progress info
        if (itr+1) % FREQ == 0:
            torch.save(Qnet.state_dict(), os.path.join(dn, 'Qnet%06d.pth' % (itr+1, )))
            print('iteration: ', itr+1, flush=True)
            with open(os.path.join(dn, 'reward.pkl'), 'wb') as f:
                pickle.dump(R, f)
    print()
    print('trainning is done.')