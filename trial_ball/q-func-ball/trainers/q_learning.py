import sys
from collections import deque
import random
import numpy as np
import warnings
from trainer import Trainer


class QTrainer(Trainer):
    def __init__(self, env, agent, BORDER):
        super().__init__(env, agent, BORDER)

    def train(self, episode_count=10000, gamma=0.9, lr=0.1, report_interval=500, verbose=True):
        if verbose:
            print(f'train episode num = {episode_count}')
            print(f'display {episode_count // report_interval} times')

        for e in range(episode_count):
            s = self.env.reset()
            d = self.env.done
            step_count = 0
            reward_sum = 0
            while not d:
                a = self.agent.policy(s)
                n_state, reward, d = self.env.step(a)
                self.update_function(s, a, n_state, reward, d, gamma, lr)
                s = n_state
                step_count += 1
                reward_sum += reward
            # 直近のエピソードの報酬の総和を記録
            self.agent.logger.log(reward_sum, step_count)

            if verbose and e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)
        return self.agent
    
    def update_function(self, s, a, ns, reward, d, gamma, lr):
        '''集めた経験を使ってAgentの行動評価関数を更新する'''
        # 勾配降下法でパラメーターQを更新
        u = np.zeros(self.env.action_num)
        r = reward
        if not d:
            r += gamma * max(self.agent.Q(ns))
        u[a] = r
        TD = s.vec @ s.vec.T * self.agent._Q.Q - (np.diag(u) @ np.vstack([s.vec for _ in range(u.size)])).T
        self.agent._Q.Q -= lr * TD

