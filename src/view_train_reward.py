import os
import pickle
import numpy as np

def moving_average(r, n=100):
    print(n)
    ret = np.cumsum(r)
    ret[n:] = ret[n:] - ret[:-n]
    moving_average = ret[n - 1:] / n
    return moving_average


def show_reward(ax, dir_path, n=1000, label='', head=None, mono=True):
    if mono:
        fig, ax = plt.add_subplot(111)
    dir_path = os.path.abspath(dir_path)
    
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Average Reward')
    with open(os.path.join(dir_path, 'reward.pkl'), 'rb') as f:
        reward = pickle.load(f)
    ma = moving_average(reward, n=n)
    ax.plot(ma, label=label)
    ax.legend(bbox_to_anchor=(-0.5, 0), loc='lower left')
    if mono:
        plt.show()
    return ma
    