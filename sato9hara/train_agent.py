import os
import argparse
import pickle
import numpy as np
from numpy.fft import fftshift, fft2
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal
from skimage import restoration

import torch
from torch import nn
from torch import optim
from torchvision import datasets

### model ###
class MnistNet(torch.nn.Module):
    def __init__(self, c=1, m=[20, 20, 5]):
        super(MnistNet, self).__init__()
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
    
### functions ###
def blur(x, kernel, c=3):
    for i in range(c):
        x = fftconvolve(x, kernel, mode='same')
    return x

def get_img(imgs, blurred_imgs, channel=1):
    n = blurred_imgs.shape[0]
    i = np.random.choice(n, 1)
    y, x = imgs[i], blurred_imgs[i]
    if channel == 1:
        x = x.unsqueeze(0)
    elif channel == 2:
        x = torch.cat([x, x], dim=0).unsqueeze(0)
    return (x, y.unsqueeze(0))

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
            q, _ = Qnet(x.to(device))
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
def train(channel=1, weight=0.0, outdir='./', gpu=0, seed=0):
    device = 'cuda:%d' % (gpu,)

    # result directory & file
    dn = '%schannel%02d_weight%03d_seed%02d' % (outdir, channel, int(100*weight), seed)
    os.makedirs(dn, exist_ok=True)

    # kernels
    std = 3.0
    f = np.vectorize(lambda x, y: multivariate_normal([0.0, 0.0], np.diag([std]*2)).pdf([x, y]))
    X, Y = np.meshgrid(np.arange(-2, 3, 1, dtype=np.float32), np.arange(-2, 3, 1, dtype=np.float32))
    kernel1 = f(X, Y)
    kernel1 = kernel1 / kernel1.sum()
    np.random.seed(0)
    kernel2 = np.random.rand(*kernel1.shape)
    kernel2 = kernel2 / kernel2.sum()

    # filters
    filt = []
    filt.append(lambda x: x)
#     filt.append(lambda x: np.maximum(0, fftconvolve(x, kernel1, mode='same')))
#     filt.append(lambda x: np.maximum(0, fftconvolve(x, kernel2, mode='same')))
    filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
    filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel2, 1e-2)))

    # images
    data = datasets.MNIST(root='./data', train=True, download=True)
    
    np.random.seed(seed)
    idx = np.random.choice(data.data.shape[0], 1000)
    imgs = data.data[idx].numpy() / 255
#     blurred_imgs = np.stack(list(map(lambda x: blur(x, kernel1, c=3), imgs)), axis=0)
    blurred_imgs = list(map(lambda x: blur(x, kernel1, c=3), imgs[:500])) + list(map(lambda x: blur(x, kernel2, c=3), imgs[500:]))
    blurred_imgs = np.stack(blurred_imgs, axis=0)
    imgs = torch.from_numpy(imgs).to(torch.float)
    blurred_imgs = torch.from_numpy(blurred_imgs).to(torch.float)

    # setup
    torch.manual_seed(seed)
    Qnet = MnistNet(c=channel, m=[20, 20, len(filt)]).to(device)
    
    # set parameters
#     Qnet.load_state_dict(torch.load('withRandom/channel01_weight000_seed00/Qnet020000.pth'))
    loss_fn = torch.nn.MSELoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Qnet.parameters(), lr=1e-3, momentum=0.9)

    # parameters
    eps = 0.1
    gamma = 0.9
    batch = 50

    # training
    R = []
    history = []
    for itr in range(40000):
        np.random.seed(itr)
        for _ in range(20):
            Ri = 0
            s = get_img(imgs, blurred_imgs, channel)
            for i in range(5):
                s, r, history = step(Qnet, s, history, filt, channel, gamma=gamma, eps=eps, device=device)
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
                X = np.stack([f(xn) for f in filt], axis=0)
                err = np.sum((X - yn)**2, axis=(1, 2))
                b = np.argmin(err)
                t.append(b)
            t = torch.tensor(t)

        x, q, t = x.to(device), q.to(device), t.to(device)
        z, _ = Qnet(x)
        loss1 = 0
        for b in range(len(filt)):
            idx = np.where(a == b)[0]
            if idx.size == 0:
                continue
            loss1 = loss1 + loss_fn(z[idx, b], q[idx])
        loss2 = loss_fn2(z, t)
        loss = loss1 + weight * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save
        if (itr+1) % 500 == 0:
            torch.save(Qnet.state_dict(), '%s/Qnet%06d.pth' % (dn, itr+1))
            with open('%s/reward.pkl' % (dn,), 'wb') as f:
                pickle.dump(R, f)

### main ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent & Save')
    parser.add_argument('--channel', default=1, type=int, help='number of channels')
    parser.add_argument('--weight', default=0, type=float, help='weight of loss2')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=0, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    args = parser.parse_args()
    assert args.channel in [1, 2]
    for seed in range(args.start, args.end):
        train(channel=args.channel, weight=args.weight, outdir=args.outdir, gpu=args.gpu, seed=seed)
        
