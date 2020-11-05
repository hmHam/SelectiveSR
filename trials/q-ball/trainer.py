import numpy as np


class Trainer(object):
    def __init__(self, env, agent, BORDER):
        self.border = BORDER
        self.env = env
        self.agent = agent

    def train(
        self,
        episode_count=1000,
        gamma=0.9,
        learning_rate=0.05,
        report_interval=500,
        verbose=True
    ):
        self.agent.logger.reset()
        self.agent.initialize_Q_table(self.env)
        # TODO: stateを意味のある情報でつくる
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            while not done:
                a = self.agent.policy(s, len(self.env.actions))
                n_state, reward, done = self.env.step(a)
                
                n_action = self.agent.policy(n_state, len(self.env.actions))
                gain = reward + gamma * self.agent.Q[n_state.x][n_action]
                estimated = self.agent.Q[s.x][a]
                self.agent.Q[s.x][a] += learning_rate * (gain - estimated)
                s = n_state
            # self.agent.Qの0 - 10以外の学習結果は削除する
            for out_state in [k for k in self.agent.Q if k < 0 or k > self.border]:
                del self.agent.Q[out_state]
            self.agent.logger.log(reward, self.env.step_count)

            # 学習の進捗状況を表示
            if verbose and e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.agent.logger.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
            episode, mean, std))
