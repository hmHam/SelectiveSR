import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import dilig
import numpy as np


# 可逆シフト
S = 8
# S = 6
FUNCS_INVERT = [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in [(0, S),(0, -S),(-S, 0),(S, 0)]
]
ACTIONS_INVERT = [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in [(0, 2), (0, -2), (-2, 0), (2, 0)]
]

# 不可逆変換
def shift(A, dx, dy):
    if dx != 0 and dy != 0:
        raise ValueError('dx dy おかしい')
    if dx == 0:
        if dy > 0: #上
            A[-dy:A.shape[0], :] = 0
        else: #下
            A[0:abs(dy), :] = 0
    else:
        if dx > 0: #右
            A[:, -dx:A.shape[1]] = 0
        else:  #左
            A[:, 0:abs(dx)] = 0
    return np.roll(A, (dy, dx), axis=(0, 1))
    

# テスト用の対角要素も含んだ変換の候補
FUNCS_DIAG = [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in [
        (0, S),(0, -S),(-S, 0),(S, 0), (S, S),(-S, -S),(S, -S), (-S, S)
    ]
]

# 増やしていく行動集合
np.random.seed(0)
n = 20
ACTIONS_DISASTER = ACTIONS_INVERT + [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in np.random.randint(3, 10, (n, 2))
]