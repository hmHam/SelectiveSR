from collections import defaultdict, namedtuple
import numpy as np
from trainer import Trainer

Experience = namedtuple('Experience', ('s', 'a', 'n_s', 'r', 'd'))

class MonteCalroTrainer(Trainer):
    def train(self, episode_count=1000, gamma=0.9, report_interval=500, verbose=True):
        self.agent.logger.reset()
        N = defaultdict(lambda: [0] * self.env.action_num)
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            experience = []
            while not done:
                a = self.agent.policy(s)
                n_state, reward, done = self.env.step(a)

                experience.append(
                    Experience(s, a, n_state, reward, done)
                )
                s = n_state
                # 学習の進捗状況をターミナル上で表示
                if verbose and e != 0 and e % report_interval == 0:
                    self.progress_report(episode=e, interval=report_interval)

            # Q値を更新
            for i, e in enumerate(experience):
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += pow(gamma, t) * experience[j].r
                    t += 1
                # ある状態で何度も行動を取る経験はあまり学習しないようにしていく
                N[e.s][e.a] += 1
                alpha = 1 / N[e.s][e.a]
                estimated = self.agent._Q[e.s][e.a]
                self.agent._Q[e.s][e.a] += alpha * (G - estimated)

            self.agent.logger.log(reward, self.env.step_count)
