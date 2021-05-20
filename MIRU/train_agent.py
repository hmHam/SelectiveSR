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
from torchvision import datasets, transforms


### data preparation ###
class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, transform=None):
        self.source = source
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return self.transform(self.source[idx]).to(torch.float)
    

def augument_data(source):
    data = []
    for c in source:
        data.extend([
            c,
            np.rot90(c, np.random.randint(1, 4)),
            np.fliplr(c),
            np.flipud(c)
        ])
    data = np.array(data)
    return data


def blur(x, kernel, c=3):
    for i in range(c):
        x = fftconvolve(x, kernel, mode='same')
    return x


def get_gauss_filt(sigma, size=5):
    f = np.vectorize(lambda x, y: multivariate_normal([0.0, 0.0], sigma).pdf([x, y]))
    X, Y = np.meshgrid(np.arange(-(size//2), size//2+1, 1, dtype=np.float32), np.arange(-(size//2), size//2+1, 1, dtype=np.float32))
    kernel = f(X, Y)
    return kernel / kernel.sum()

    
def load_data(seed=0):
    # blur filters
    SIZE = 9
    kernel1 = get_gauss_filt(np.diag([10**2, 10**2]), size=SIZE)
    kernel2 = get_gauss_filt(np.diag([10**2, 1]), size=SIZE)
    kernel3 = get_gauss_filt(np.diag([1, 10**2]), size=SIZE)
    blur_filters = [kernel1, kernel2, kernel3]
    train_blur_filter = kernel1
    
    if os.path.exists('data.npz'):
        data = np.load('data.npz')
        imgs = data['imgs']
        blurred_imgs = data['blurred_imgs']
    else:
        # images
        print('create blurred imgs...')
        SAMPLE_SIZE = 1000
        data = datasets.MNIST(root='./data', train=True, download=True)
        np.random.seed(seed)
        idx = np.random.choice(data.data.shape[0], SAMPLE_SIZE)
        RESOLUTION = (250, 250)
        imgs = []
        for i in idx:
            img, _  = data[i]
            img = img.resize(RESOLUTION)
            img = np.asarray(img)/255
            imgs.append(img)
        imgs = augument_data(imgs)
        blurred_imgs = np.stack(list(map(lambda x: blur(x, train_blur_filter, c=3), imgs)), axis=0)
        # 3次元に
        imgs = imgs[..., None]
        blurred_imgs = blurred_imgs[..., None]
        np.savez('data.npz', imgs=imgs, blurred_imgs=blurred_imgs)
        print('save datasets!')
    imgs = Dataset(imgs)
    blurred_imgs = Dataset(blurred_imgs)
    return imgs, blurred_imgs, blur_filters


def get_img(imgs, blurred_imgs, channel=1):
    n = len(blurred_imgs)
    i = np.random.choice(n)
    y, x = imgs[i], blurred_imgs[i]
    if channel == 1:
        x = x.unsqueeze(0)
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
            torch.nn.Conv2d(c, self.m[0], 15, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(m[0]),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(self.m[0], self.m[1], 15, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(m[1]),
            torch.nn.MaxPool2d(2, stride=2)
        )
        self.OH = 52
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
def train(data, channel=1, weight=0.0, outdir='./', gpu=0, seed=0):
    print('start preparation...')
    device = 'cuda:%d' % (gpu,)

    # result directory & file
    dn = '%schannel%02d_weight%03d_seed%02d' % (outdir, channel, int(100*weight), seed)
    os.makedirs(dn, exist_ok=True)

    imgs, blurred_imgs, blur_filters = data
    kernel1, kernel2, kernel3 = blur_filters
    # actions
    filt = []
    filt.append(lambda x: x)
    filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
    filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
    filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))


    # setup
    torch.manual_seed(seed)
    Qnet = QNet(c=channel, m=[20, 20, len(filt)]).to(device)
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
            s = get_img(imgs, blurred_imgs, channel)
            for i in range(5):
                s, r, history = step(Qnet, s, history, filt, channel, gamma=gamma, eps=eps, device=device)
                Ri = Ri + r
            R.append(Ri)

        if len(history) < batch:
            continue

        print('hoge', len(history))
        with torch.no_grad():
            idx = np.random.choice(len(history), batch)
            x, x_next, y, a, r = subhistory(Qnet, history, idx)
            print(itr, end='/')

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
        z, p = Qnet(x)
        loss1 = 0
        for b in range(len(filt)):
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
    parser.add_argument('--channel', default=1, type=int, help='number of channels')
    parser.add_argument('--weight', default=0, type=float, help='weight of loss2')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--start', default=0, type=int, help='start seed')
    parser.add_argument('--end', default=0, type=int, help='end seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    args = parser.parse_args()
    assert args.channel in [1, 2]
    for seed in range(args.start, args.end):
        data = load_data(seed=seed)
        train(data, channel=args.channel, weight=args.weight, outdir=args.outdir, gpu=args.gpu, seed=seed)
        
