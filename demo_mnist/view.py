from matplotlib import pyplot as plt
import torch as tc

def view_moving_average(r, n=100):
    ret = tc.cumsum(r, dim=0, dtype=tc.float)
    ret[n:] = ret[n:] - ret[:-n]
    moving_average = ret[n - 1:] / n
    # view on matplotlib
    plt.plot(moving_average, label='got reward while training')
    plt.legend(loc='best')