import sys
from collections import deque
import random
import numpy as np
from agent import Experience
import warnings

class Observer(object):
    # 最初1024サンプルの経験データを使用して正規化する
    def __init__(self, states):
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0)

    def transform(self, states):
        return (states - self.mean) / self.std


class Trainer(object):
    def __init__(self, env, agent, BORDER, buffer_size=1024, batch_size=32, gamma=0.9):
        # experience-replayを実装 <- 経験の偏りを防ぎ学習を安定化する
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences = deque(maxlen=batch_size)

        self.gamma = gamma
        self.border = BORDER
        self.env = env
        self.agent = agent
        self.observer = None

    def train(
        self,
        episode_count=10000,
        learning_rate=0.05,
        interval=0
    ):
        self.experiences = deque(maxlen=self.buffer_size)
        self.trained_count = 0
        self.reward_log = []
        
        # Experienceを集める
        for _ in range(episode_count):
            s = self.env.reset()
            done = self.env.done
            step_count = 0
            while not done:
                a = self.agent.policy(s, len(self.env.actions))
                n_state, reward, done = self.env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)
                # 経験のサンプル数がbuffer_sizeを超えたら更新
                if len(self.experiences) == self.buffer_size:
                    # agentの行動評価関数を更新
                    self.observer = Observer(np.vstack([e.s.val for e in self.experiences]))
                    self.update_function(learning_rate)
                s = n_state
                step_count += 1
            # 直近のエピソードの報酬の総和を記録
            reward = sum([self.experiences[i].r for i in range(len(self.experiences) - step_count, len(self.experiences))])
            self.agent.logger.log(reward, step_count)
        return self.agent
    
    def update_function(self, learning_rate):
        '''集めた経験を使ってAgentの行動評価関数を更新する'''
        sampled_experiences = random.sample(self.experiences, self.batch_size)
        # TDをつくる
        states = []
        # 各状態での実際の価値を求める
        gains = []
        for e in sampled_experiences:
            states.append(e.s)
            reward = e.r
            n_s_val = self.observer.transform(e.n_s.val)
            n_a_evals = self.agent.q_func(n_s_val)
            if not e.d:
                # On-Policyだとここが変わる
                reward += self.gamma * np.max(n_a_evals)
            n_a_evals[e.a] = reward
            gains.append(n_a_evals)
        gains = np.vstack(gains)

        # 勾配降下方でパラメーターQを更新
        for idx in range(gains.shape[0]):
            s = self.observer.transform(states[idx].val)
            gain = gains[idx]
            
            # self.agent.q_func.Q -= learning_rate * (
            #     s.T @ s * self.agent.q_func.Q - np.vstack([s] * gain.size).T @ np.diag(gain)
            # )
            for j in range(self.agent.q_func.Q.shape[1]):
                self.agent.q_func.Q[:, j] -= learning_rate * s * (s.T @ self.agent.q_func.Q[:, j] - gain[j])