'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える
from enum import IntEnum, auto
import inspect
from pathlib import Path
import argparse
import pickle
from datetime import datetime
from collections import defaultdict
from random import randint

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

TARGET_NUM = 5
BORDER = 10

class Action(IntEnum):
    PLUS_ONE = 0
    TIMES_TWO = 1
    MINUS_ONE = 2
    DIVIDE_TWO = 3
    END = 4
  

class State(object):
    # TODO: Q学習がうまくいかない
    # solution: 
    # (1) 偶数か奇数かなど xをいくつかの細かい情報にわける -> ベクトル化
    # (2) Q-tableをNNなどで近似 <- 分散表現が
    def __init__(self, x):
        self.x = x
    
    def clone(self):
        return State(self.x)


class Env(object):
    '''報酬を返す環境'''
    def __init__(self):
        self.agent_state = State(randint(0, BORDER))
        self.done = False
        self.step_count = 0

    def reset(self):
        # REVIEW: おかしいかもしれん
        self.agent_state = State(randint(0, BORDER))
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
        return - BORDER / 5 + (BORDER- abs(state.x - TARGET_NUM))

    def _move(self, state, action):
        s = state.clone()
        if state.x == TARGET_NUM:
            self.done = True
            return s
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
        # s.xが範囲を超えていたら終了して値を更新しない
        if s.x > BORDER or s.x < 0:
            self.done = True
            s = state.clone()
        return s

    def step(self, action):
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
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
            if s.x in self.Q and sum(self.Q[s.x]) != 0:
                return np.argmax(self.Q[s.x])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []
        self.step_count_log = []

    def log(self, reward, step_count):
        self.reward_log.append(reward)
        self.step_count_log.append(step_count)


    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=500):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(env.actions))
        # TODO: stateを意味のある情報でつくる
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
                self.log(reward, env.step_count)

            if e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
                episode, mean, std))
        
    def plot_result(self, env, log_path='', interval=500):
        if log_path:
            fname, ext = "".join(log_path.split('.')[:-1]), log_path.split('.')[-1]
        fig1 = plt.figure(figsize=(100, 6))
        # 学習したQ-tableの値を表示
        self._plot_q_value(
            env,
            fig1.add_subplot(1, 1, 1)
        )
        if log_path:
            fig1.savefig(f'{fname}1.{ext}')
        else:
            plt.show() 

        fig2 = plt.figure(figsize=(14, 6))
        plt.subplots_adjust(hspace=0.6)
        
        # 学習段階での各エピソードの最終ステップの即時報酬をプロット
        self._plot_last_reward(
            fig2.add_subplot(2, 1, 1),
            interval
        )
        # ステップカウントの推移をプロット
        self._plot_step_count(
            fig2.add_subplot(2, 1, 2),
            interval
        )
        if log_path:
            fig2.savefig(f'{fname}2.{ext}')
        else:
            plt.show()
        
    def _plot_step_count(self, ax, interval):
        '''インターバル事のエピソードの平均ステップ数を表示'''
        indices = list(range(0, len(self.step_count_log), interval))
        step_counts = np.array([self.step_count_log[i:i+interval] for i in indices])
        means = step_counts.mean(axis=1)
        stds = step_counts.std(axis=1)

        ax.set_title('Step Count History')
        ax.grid()

        ax.fill_between(indices, means - stds, means + stds,
                            alpha=0.1, color="b")
        ax.plot(indices, means, "o-", color="b",
                    label=f"Step Counts for each {interval} episode")

    def _plot_last_reward(self, ax, interval):
        '''学習段階での各エピソードの最終ステップの即時報酬をプロット'''
        # プロットでも再利用するので先に変数化
        indices = list(range(0, len(self.reward_log), interval))
        rewards = np.array([self.reward_log[i:i+interval] for i in indices])
        means = rewards.mean(axis=1)
        stds = rewards.std(axis=1)

        ax.set_title("Reward History")
        ax.grid()
        ax.fill_between(indices, means - stds, means + stds,
                            alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g",
                    label=f"Rewards for each {interval} episode")
        ax.legend(loc="best")
        
    def _plot_q_value(self, env, ax):
        '''学習したQ-tableの値を表示'''
        vmin, vmax = min(self.Q), max(self.Q)
        kk = []
        for vals in self.Q.values():
            kk.extend(vals)
        qmin, qmax = min(kk), max(kk)
        state_num = (vmax - vmin + 1)
        size = state_num * 3
        reward_map = np.zeros((150, size))
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
        ax.set_xticks(np.arange(0, size, 3))
        ax.set_yticks([2])
        ax.set_xticklabels(np.arange(0, state_num))
        ax.set_yticklabels(np.arange(0, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_count', '-e', type=int, default=10000)
    parser.add_argument('--interval', '-i', type=int, default=50)
    parser.add_argument('--epsilon', '-ep', type=float, default=0.2)
    args = parser.parse_args()

    agent = BallAgent(epsilon=args.epsilon)
    env = Env()
    agent.learn(
        env,
        episode_count=args.episode_count,
        report_interval=args.interval
    )
    # 以下結果の表示
    timestamp = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
    log_dir = Path(f'logs/{timestamp}')
    log_dir.mkdir(exist_ok=True)

    # 報酬関数のコードを保存
    with open(log_dir / 'func.txt', 'w') as f:
        f.write(
            inspect.getsource(env.reward_func)
        )
    # ログをpickleデータとして保存
    with open(log_dir / 'q.pickle', 'wb') as f:
        pickle.dump(dict(agent.Q), f)
    # グラフを画像で保存
    agent.plot_result(env, str(log_dir / 'fig.png'))

    # グラフを表示
    agent.plot_result(env)

    # テストフェーズ