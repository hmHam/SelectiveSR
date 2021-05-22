import numpy as np

def get_funcs(delta_x, delta_y):
    func_shift = lambda A: np.roll(A, (delta_x, delta_y), axis=(0, 1))
    func_unshift = lambda A: np.roll(A, (-delta_x, -delta_y), axis=(0, 1))
    return func_shift, func_unshift