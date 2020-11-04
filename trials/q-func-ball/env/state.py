import numpy as np


class State(object):
    def __init__(self, x):
        self.x = x
        self.is_less_than_3 = x <= 3
        self.is_between_3nd8 = 3 < x <= 7
        self.is_larger_than_8 = 7 < x

    @property
    def size(self):
        '''ベクトルのサイズを返す'''
        return 4

    @property
    def val(self):
        return np.array([
            self.x,
            self.is_less_than_3,
            self.is_between_3nd8,
            self.is_larger_than_8
        ])

    def clone(self):
        return State(self.x)

