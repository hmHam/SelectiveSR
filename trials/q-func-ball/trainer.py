import sys
from collections import deque
import random
import numpy as np
import warnings


class Trainer(object):
    def __init__(self, env, agent, BORDER, gamma=0.9, learning_rate=0.05):
        self.gamma = gamma # 割引率
        self.border = BORDER
        self.env = env
        self.agent = agent
        self.learning_rate = learning_rate
        self.observer = None # 環境のベクトルを正規化する↑

    def train(self, episode_count=10000, report_interval=500, verbose=True):
        if verbose:
            print(f'train episode num = {episode_count}')
            print(f'display {episode_count // report_interval} times')
        for e in range(episode_count):
            s = self.env.reset()
            done = self.env.done
            step_count = 0
            reward_sum = 0
            while not done:
                a = self.agent.policy(s, len(self.env.actions))
                n_state, reward, done = self.env.step(a)
                # FIXME: 更新に必要な値はメソッドに渡すように修正
                self.update_function(
                    s,
                    a,
                    n_state,
                    reward,
                    done,
                    len(self.env.actions)
                )
                s = n_state
                step_count += 1
                reward_sum += reward
            # 直近のエピソードの報酬の総和を記録
            self.agent.logger.log(reward_sum, step_count)

            if verbose and e != 0 and e % report_interval == 0:
                self.progress_report(episode_count, episode_index=e, interval=report_interval)
        return self.agent
    
    def update_function(self, s, a, ns, reward, d, action_size):
        '''集めた経験を使ってAgentの行動評価関数を更新する'''
        # 勾配降下法でパラメーターQを更新
        u = np.zeros(action_size)
        r = reward
        if not d:
            r += self.gamma * max(self.agent.q_func(ns.vec))
        u[a] = r
        TD = s.vec @ s.vec.T * self.agent.q_func.Q - (np.diag(u) @ np.vstack([s.vec for _ in range(u.size)])).T
        self.agent.q_func.Q -= self.learning_rate * TD

    def progress_report(self, episode_count, episode_index, interval=100):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.agent.logger.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print(f"At Episode {episode_index} average reward is {mean} (+/-{std}).", end='\t')
        print(f'{episode_count // interval - episode_index // interval} times left')
