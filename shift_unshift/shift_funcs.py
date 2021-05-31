import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import dilig
import numpy as np


# FUNCS, ACTIONSをセットで渡す場合
# NOTE: こいつは没になったが一応残している。
def get_funcs(
    dx,
    dy
):
    func_shift = dilig(lambda A: np.roll(A, (dx, dy), axis=(0, 1))) 
    func_unshift = dilig(lambda A: np.roll(A, (-dx, -dy), axis=(0, 1)))
    return func_shift, func_unshift

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

# 不可逆シフト 上下左右
FUNCS_IRREV = [
    dilig(lambda A, dx=dx, dy=dy: shift(A, dx, dy)) for dx, dy in [(0, S),(0, -S),(-S, 0),(S, 0)]
]
ACTIONS_IRREV = [
    dilig(lambda A, dx=dx, dy=dy: shift(A, dx, dy)) for dx, dy in [(0, 2), (0, -2), (-2, 0), (2, 0)]
]
