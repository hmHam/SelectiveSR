from collections import defaultdict, namedtuple
import numpy as np

Experience = namedtuple('Experience', ('s', 'a', 'n_s', 'r', 'd'))

class Trainer(object):
    def __init__(self, env, agent, BORDER):
        self.border = BORDER
        self.env = env
        self.agent = agent

    def train(
        self,
        episode_count=1000,
        gamma=0.9,
        report_interval=500,
        verbose=True
    ):
        self.agent.logger.reset()
        self.agent.initialize_Q_table(self.env)
        N = defaultdict(lambda: [0] * len(self.env.actions))
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            experience = []
            while not done:
                a = self.agent.policy(s, len(self.env.actions))
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
                N[e.s.x][e.a] += 1
                alpha = 1 / N[e.s.x][e.a]
                estimated = self.agent.Q[e.s.x][e.a]
                self.agent.Q[e.s.x][e.a] += alpha * (G - estimated)
            # self.agent.Qの0 - 10以外の学習結果は削除する
            for out_state in [k for k in self.agent.Q if k < 0 or k > self.border]:
                del self.agent.Q[out_state]
            self.agent.logger.log(reward, self.env.step_count)

    def progress_report(self, interval=100, episode=-1):
        '''学習の結果得られた報酬の履歴を可視化'''
        rewards = self.agent.logger.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
            episode, mean, std))
