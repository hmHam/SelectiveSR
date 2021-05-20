import numpy as np

class Trainer(object):
    def __init__(self, env, agent, BORDER):
        self.border = BORDER
        self.env = env
        self.agent = agent

    def train(self):
        raise NotImplementedError('plz implement train method')

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.agent.logger.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
            episode, mean, std))
