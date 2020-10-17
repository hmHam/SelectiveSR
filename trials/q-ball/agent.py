from collections import defaultdict

import numpy as np

from logger import Logger


class BallAgent(object):
    '''位置を特定のポリシーに合わせて更新していく'''
    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.logger = Logger(self)

    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s.x in self.Q and sum(self.Q[s.x]) != 0:
                return np.argmax(self.Q[s.x])
            else:
                return np.random.randint(len(actions))

    def learn(
        self,
        env,
        episode_count=1000,
        gamma=0.9,
        learning_rate=0.1,
        report_interval=500,
        verbose=True
    ):
        self.logger.reset()
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
                self.logger.log(reward, env.step_count)

            # 学習の進捗状況を表示
            if verbose and e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.logger.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
                episode, mean, std))
