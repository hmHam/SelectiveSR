import numpy as np

from logger import Logger


class Agent(object):
    def __init__(self, env, epsilon):
        self.epsilon = epsilon
        self.logger = Logger(self)
        self.env = env

    def Q(self, s):
        raise NotImplementedError("plz implement Q function")

    def policy(self, s, play=False):
        if not play and np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_num)
        # Q-tableに情報が存在する場合
        return np.argmax(self.Q(s))
        
    def play(self, env, x):
        state = env.reset(x)
        done = False
        reward_sum = 0
        step_count = 0
        while not done:
            action = self.policy(state, play=True)
            n_state, reward, done = env.step(action)
            reward_sum += reward
            state = n_state
            step_count += 1
        return {
            'total_reward': reward_sum,
            'step_count': step_count
        }