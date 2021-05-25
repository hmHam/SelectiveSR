import sys
import os

import numpy as np
from skimage import restoration

sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import dilig


def get_gauss_filt(sigma, size=5):
    f = np.vectorize(lambda x, y: multivariate_normal([0.0, 0.0], sigma).pdf([x, y]))
    X, Y = np.meshgrid(np.arange(-(size//2), size//2+1, 1, dtype=np.float32), np.arange(-(size//2), size//2+1, 1, dtype=np.float32))
    kernel = f(X, Y)
    return kernel / kernel.sum()


def blur(x, kernel, c=3):
    for i in range(c):
        x = fftconvolve(x, kernel, mode='same')
    return x


def get_funcs(sigma, size):
    kernel = get_gauss_filt(sigma, size=size)
    func = dilig(lambda x: blur(x, kernel))
    inv = dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel, 1e-2)))
    return func, inv