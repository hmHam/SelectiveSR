import numpy as np


class State(object):
    def __init__(self, x, BORDER):
        self.border = BORDER
        # FIXME: one-hot
        self.x = x
        self.is_less_than_3 = x <= 3
        self.is_between_3nd8 = 3 < x <= 7
        self.is_larger_than_8 = 7 < x

    @property
    def size(self):
        '''ベクトルのサイズを返す'''
        return self.vec.size

    @property
    def vec(self):
        one_hot_x = np.zeros(self.border + 1)
        if 0 <= self.x <= self.border:
            one_hot_x[self.x] = 1
        return np.r_[
            one_hot_x,
            np.array([
                self.is_less_than_3,
                self.is_between_3nd8,
                self.is_larger_than_8
            ])
        ]

    def clone(self):
        return State(self.x, self.border)

