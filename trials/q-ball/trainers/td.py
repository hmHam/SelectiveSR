from trainer import Trainer


class TDTrainer(Trainer):
    def train(self, episode_count=1000, gamma=0.9, lr=0.1, report_interval=500, verbose=True):
        self.agent.logger.reset()
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            while not done:
                a = self.agent.policy(s)
                s_next, reward, done = self.env.step(a)
                self.update(self.agent, s, a, reward, s_next, gamma, lr)
                s = s_next
            self.agent.logger.log(reward, self.env.step_count)

            # 学習の進捗状況を表示
            if verbose and e != 0 and e % report_interval == 0:
                self.progress_report(episode=e, interval=report_interval)

    def update(self, agent, s, a, reward, s_next, gamma, lr):
        raise NotImplementedError('plz implement update method')
