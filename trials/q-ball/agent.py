from collections import defaultdict
import numpy as np

from logger import Logger


class BallAgent(object):
    '''位置を特定のポリシーに合わせて更新していく'''
    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.logger = Logger(self)

    def initialize_Q_table(self, env):
        # TODO: __init__に入れれるか検討
        self.Q = defaultdict(lambda: [0] * len(env.actions))

    def policy(self, s, action_num):
        if np.random.random() < self.epsilon:
            return np.random.randint(action_num)
        # Q-tableに情報が存在する場合
        if s.x in self.Q and sum(self.Q[s.x]) != 0:
            return np.argmax(self.Q[s.x])
        return np.random.randint(action_num)
    
    def play(self, env, x):
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