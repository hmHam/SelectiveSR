import sys
import os

import numpy as np
from skimage import restoration
from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
)
from src import dilig


def get_gauss_filt(Sigma, size=5):
    f = np.vectorize(lambda x, y: multivariate_normal([0.0, 0.0], Sigma).pdf([x, y]))
    X, Y = np.meshgrid(np.arange(-(size//2), size//2+1, 1, dtype=np.float32), np.arange(-(size//2), size//2+1, 1, dtype=np.float32))
    kernel = f(X, Y)
    return kernel / kernel.sum()


def blur(x, kernel, c=3):
    for i in range(c):
        x = fftconvolve(x, kernel, mode='same')
    return x

# まとめてGaussFilterだけを生成する場合
def get_funcs(sigma, size):
    kernel = get_gauss_filt(sigma, size=size)
    func = dilig(lambda x: blur(x, kernel))
    inv = dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel, 1e-2)))
    return func, inv


### 実験1
FUNCS_GR = []
std = 3.0
kernel1 = get_gauss_filt(np.diag([std]*2))
# kernel1 = get_gauss_filt(np.diag([std]*2))  #sigma = \sqrt{3.0}
kernel1 = kernel1 / kernel1.sum()
FUNCS_GR.append(
    dilig(lambda x, c=3: blur(x, kernel1, c))
)
np.random.seed(0)
kernel2 = np.random.rand(*kernel1.shape)
kernel2 = kernel2 / kernel2.sum()
FUNCS_GR.append(
    dilig(lambda x, c=3: blur(x, kernel2, c))
)
ACTIONS_GR = []
ACTIONS_GR.append(
    dilig(lambda x: np.maximum(0, fftconvolve(x, kernel1, mode='same')))
)
ACTIONS_GR.append(
    dilig(lambda x: np.maximum(0, fftconvolve(x, kernel2, mode='same')))
)
ACTIONS_GR.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
)
ACTIONS_GR.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel2, 1e-2)))
)

### 実験2
# ランダムフィルタでぼやかす処理を候補から抜く
FUNCS_WO_RN_BLUR = FUNCS_GR

ACTIONS_WO_RN_BLUR = []
ACTIONS_WO_RN_BLUR.append(
    dilig(lambda x: np.maximum(0, fftconvolve(x, kernel1, mode='same')))
)
ACTIONS_WO_RN_BLUR.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
)
ACTIONS_WO_RN_BLUR.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel2, 1e-2)))
)

### 実験3
# ランダムオンパレード
# kernel3.png参照
a = np.ones(kernel1.shape)
a[1:4, 1:4] = 0

kernel3 = np.ones(kernel1.shape)
kernel3[1:3, 1:3] = -3
kernel3[2, 2] = 5
kernel3 = kernel3 + a
kernel3 = kernel3 / kernel3.sum()

kernel4 = np.ones(kernel1.shape)
kernel4[3:5, 3:5] = 2
kernel4[1:3, 1:3] = -5
kernel4 = kernel4 + 3 * a
kernel4 = kernel4 / kernel4.sum()

FUNCS_RN = FUNCS_GR + [
    dilig(lambda x, c=3: blur(x, kernel3, c)),
    dilig(lambda x, c=3: blur(x, kernel4, c))
]

ACTIONS_RN = []
ACTIONS_RN.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
)
ACTIONS_RN.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel2, 1e-2)))
)
ACTIONS_RN.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel3, 1e-2)))
)
ACTIONS_RN.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel4, 1e-2)))
)


### ACTIONS_GR2, kernel2を変える
ACTIONS_GR2 = []
ACTIONS_GR2.append(
    dilig(lambda x: np.maximum(0, fftconvolve(x, kernel1, mode='same')))
)
ACTIONS_GR2.append(
    dilig(lambda x: np.maximum(0, fftconvolve(x, kernel4, mode='same')))
)
ACTIONS_GR2.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
)
ACTIONS_GR2.append(
    dilig(lambda x: np.maximum(0, restoration.wiener(x, kernel4, 1e-1)))
)