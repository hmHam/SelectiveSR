'''
重みパラメータを自動で学習する
'''

import numpy as np
from matplotlib import pyplot as plt

# 活性化関数 = 入力信号の総和を出力信号に変換する関数
# シグモイド関数 h(x) = 1/1+ exp(-x)


# 隠れ層

def step_function(x):
    return (x > 0).astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


# 出力層
def identity_function(x):
    return x


def softmax(a):
    '''
    ニューラルネットワークの学習時に使用
    推論時はこれを挟まないことが多い。
    '''
    a = a - np.max(a)  # オーバーフロー対策
    return np.exp(a) / np.sum(np.exp(a))


if __name__ == '__main__':
    X = np.array([1, 2])
    W = np.array([
        [1, 3, 5],
        [2, 4, 6],
    ])
    Y = sigmoid(np.dot(X, W))
    plt.plot(X, Y)
    plt.ylim(-0.1, 1.1)
    plt.show()
