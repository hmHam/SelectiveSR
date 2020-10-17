# coding: utf-8
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 10000 # 高速化のため
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

# float16に型変換
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))
