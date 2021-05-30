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
    delta_x,
    delta_y
):
    func_shift = dilig(lambda A: np.roll(A, (delta_x, delta_y), axis=(0, 1))) 
    func_unshift = dilig(lambda A: np.roll(A, (-delta_x, -delta_y), axis=(0, 1)))
    return func_shift, func_unshift

# 可逆シフト
# S = 8
S = 6
FUNCS_INVERT = [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in [(0, S),(0, -S),(-S, 0),(S, 0)]
]
ACTIONS_INVERT = [
    dilig(lambda A, dx=dx, dy=dy: np.roll(A, (dy, dx), axis=(0, 1))) for dx, dy in [(0, 2), (0, -2), (-2, 0), (2, 0)]
]

# 不可逆変換
def shift(A, delta_x, delta_y):
    if delta_x != 0 and delta_y != 0:
        raise ValueError('dx dy おかしい')
    if delta_x == 0:
        if delta_y > 0: #上
            A[-delta_y:A.shape[0], :] = 0
        else: #下
            A[0:abs(delta_y), :] = 0
    else:
        if delta_x > 0: #右
            A[:, -delta_x:A.shape[1]] = 0
        else:  #左
            A[:, 0:abs(delta_x)] = 0
    return np.roll(A, (delta_y, delta_x), axis=(0, 1))

def factory_irrevertible_funcs(delta_x, delta_y):
    return dilig(lambda A: shift(A, delta_x, delta_y))

def factory_irrevertible_actions(delta_x, delta_y):
    return dilig(lambda A: shift(A, delta_x, delta_y))

# 不可逆シフト 上下左右
FUNCS_IRREV = [
    factory_irrevertible_funcs(dx, dy) for dx, dy in [(0, 8),(0, -8),(-8, 0),(8, 0)]
]
ACTIONS_IRREV = [
    factory_irrevertible_actions(dx, dy) for dx, dy in [(0, 2), (0, -2), (-2, 0), (2, 0)]
]
