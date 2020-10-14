'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
from enum import IntEnum, auto
import pickle
from datetime import datetime
from collections import defaultdict
from random import randint

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

TARGET_NUM = 100

class Action(IntEnum):
    PLUS_ONE = 0
    TIMES_TWO = 1
    MINUS_ONE = 2
    DIVIDE_TWO = 3
    END = 4
  

class State(object):
    def __init__(self, x):
        self.x = x
    
    def clone(self):
        return State(self.x)


class Env(object):
    '''報酬を返す環境'''
    def __init__(self):
        self.agent_state = State(randint(0, 255))
        self.done = False
        self.step_count = 0

    def reset(self):
        # REVIEW: おかしいかもしれん
        self.agent_state = State(randint(0, 255))
        self.done = False
        self.step_count = 0
        return self.agent_state

    @property
    def actions(self):
        return [
            Action.PLUS_ONE,
            Action.TIMES_TWO,
            Action.MINUS_ONE,
            Action.DIVIDE_TWO,
            Action.END,
        ]

    def reward_func(self, state):
        # FIXME: (要)重み付け, １項目: ステップ数のペナルティ
        # ２項目100との距離が近いほど大きく
        return -50 + (255 - abs(state.x - TARGET_NUM))

    def _move(self, state, action):
        s = state.clone()
        self.step_count += 1
        if action == Action.PLUS_ONE:
            s.x += 1
        elif action == Action.TIMES_TWO:
            s.x *= 2
        elif action == Action.MINUS_ONE:
            s.x -= 1
        elif action == Action.DIVIDE_TWO:
            s.x //= 2
        elif action == Action.END:
            # 値は変更しない
            self.done = True
            self.step_count -= 1
        return s

    def step(self, action):
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
        if state.x == TARGET_NUM or state.x > 255 or state.x < 0:
            self.done = True
            next_state = state.clone()
        else:
            next_state = self._move(state, action)
        reward = self.reward_func(state)
        self.agent_state = next_state
        return next_state, reward, self.done
        

class BallAgent(object):
    '''位置を特定のポリシーに合わせて更新していく'''
    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s.x]) != 0:
                return np.argmax(self.Q[s.x])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=500):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(env.actions))
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                a = self.policy(s, env.actions)
                n_state, reward, done = env.step(a)

                gain = reward + gamma * max(self.Q[n_state.x])
                estimated = self.Q[s.x][a]
                self.Q[s.x][a] += learning_rate * (gain - estimated)
                s = n_state
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
                episode, mean, std))
    
    def plot_learned_log(self, ax, interval=500):
        indices = list(range(0, len(self.reward_log), interval))
        means = []
        stds = []
        for i in indices:
            rewards = self.reward_log[i:(i + interval)]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        means = np.array(means)
        stds = np.array(stds)
        ax.set_title("Reward History")
        ax.grid()
        ax.fill_between(indices, means - stds, means + stds,
                            alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g",
                    label="Rewards for each {} episode".format(interval))
        ax.legend(loc="best")
        
    def plot_q_value(self, env, ax):
        vmin, vmax = min(self.Q), max(self.Q)
        kk = []
        print(vmax, vmin)
        for vals in self.Q.values():
            kk.extend(vals)
        qmin, qmax = min(kk), max(kk)
        action_num = len(env.actions)
        reward_map = np.zeros((150, (vmax - vmin) * 3))
        for s, vals in self.Q.items():
            _c = 3 * s + 1
            for j in range(50, 100):
                reward_map[j][_c - 1] = 255 * vals[2] / (qmax - qmin) # -1
                reward_map[j][_c + 1] = 255 * vals[0]/ (qmax - qmin) # +1
                reward_map[j][_c] = 255 * vals[4] / (qmax - qmin) # END    
            for j in range(50):
                reward_map[j][_c] = 255 * vals[1] / (qmax - qmin)# x2
            for j in range(100, 150):
                reward_map[j][_c] = 255 * vals[3] / (qmax - qmin) # /2
        ax.axvline(x=3 * 100 + 1, ymin=qmin, ymax=qmax)
        ax.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
        
    def plot_result(self, env):
        fig = plt.figure(figsize=(14, 6))
        self.plot_q_value(env, fig.add_subplot(1, 2, 1))
        self.plot_learned_log(fig.add_subplot(1, 2, 2))
        plt.show()


if __name__ == '__main__':
    agent = BallAgent(epsilon=0.2)
    env = Env()
    agent.learn(env, episode_count=10000)
    agent.plot_result(env)
    # TODO: 適当な乱数と推論をして比較

    # ログをpickleデータとして保存
    with open('logs/' + datetime.now().strftime('%Y-%m-%d:%H:%M:%S'), 'wb') as f:
        pickle.dump(dict(agent.Q), f)