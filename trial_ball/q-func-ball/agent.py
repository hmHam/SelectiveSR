from collections import defaultdict
import numpy as np

from logger import Logger
from agent import Agent


class QFunction(object):
    def __init__(self, ns, na):
        self.Q = np.zeros((ns, na))

    def __call__(self, s):
        # 4 x (4, 5) => (5,)
        return s.vec @ self.Q


class BallAgent(Agent):
    '''位置を特定のポリシーに合わせて更新していく'''
    # 持っていて欲しい機能
    # * ε-Greedy法で動く方策
    # * 行動評価値を返す関数
    def __init__(self, env, epsilon):
        # TODO: 行列で関数を表現
        self.onPolicy = False
        self._Q = QFunction(env.B + 1, env.action_num)
        super().__init__(env, epsilon)

    def Q(self, s, a=None):
        return self._Q(s)

    def policy(self, s):
        # e-Greedy法
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_num)
        action_evals = self.Q(s)
        if self.onPolicy:
            return np.random.choice(self.env.actions, p=action_evals)
        return np.argmax(action_evals)
