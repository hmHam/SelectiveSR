import sys
import os
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from src import dilig
import numpy as np


def get_funcs(
    delta_x,
    delta_y
):
    func_shift = dilig(lambda A: np.roll(A, (delta_x, delta_y), axis=(0, 1)))
    func_unshift = dilig(lambda A: np.roll(A, (-delta_x, -delta_y), axis=(0, 1)))
    return func_shift, func_unshift