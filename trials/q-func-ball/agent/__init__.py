from collections import defaultdict, namedtuple
import numpy as np

from .logger import Logger
from .function import QFunction

# d = doneのフラグ
Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])


class BallAgent(object):
    '''位置を特定のポリシーに合わせて更新していく'''
    # 持っていて欲しい機能
    # * ε-Greedy法で動く方策
    # * 行動評価値を返す関数
    def __init__(self, epsilon, ns, actions):
        self.epsilon = epsilon
        self.actions = actions
        # TODO: 行列で関数を表現
        self.onPolicy = False
        self.q_func = QFunction(ns, len(actions))
        self.logger = Logger(self)

    def policy(self, s, action_num):
        # e-Greedy法
        if np.random.random() < self.epsilon:
            return np.random.randint(action_num)
        action_evals = self.q_func(s.val)
        if self.onPolicy:
            return np.random.choice(self.actions, p=action_evals)
        return np.argmax(action_evals)
    
    def play(self, env, x):
        '''運用時のトライアル'''
        state = env.reset(x)
        done = False
        reward_sum = 0
        step_count = 0
        while not done:
            action = self.policy(state, len(env.actions))
            n_state, reward, done = env.step(action)
            reward_sum += reward
            state = n_state
            step_count += 1
        return {
            'total_reward': reward_sum,
            'step_count': step_count
        }