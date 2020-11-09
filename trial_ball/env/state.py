import numpy as np


class State(object):
    def __init__(self, x, B):
        self.x = x
        self.B = B

    def __hash__(self):
        return hash(self.x)

    def __eq__(self, other):
        return self.x == other.x

    def clone(self):
        return State(self.x, self.B)

    @property
    def vec(self):
        one_hot_x = np.zeros(self.B + 1)
        if 0 <= self.x <= self.B:
            one_hot_x[self.x] = 1
        return one_hot_x

