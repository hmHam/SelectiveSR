import numpy as np

class QFunction(object):
    def __init__(self, ns, na):
        self.Q = np.zeros((ns, na))

    def __call__(self, s_val):
        # 4 x (4, 5) => (5,)
        return s_val @ self.Q

