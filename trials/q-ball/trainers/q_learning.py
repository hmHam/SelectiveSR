from .td import TDTrainer

class QTrainer(TDTrainer):
    def update(self, agent, s, a, reward, s_next, gamma, lr):
        gain = reward + gamma * max(agent._Q[s_next])
        estimated = agent._Q[s][a]
        agent._Q[s][a] = agent._Q[s][a] + lr * (gain - estimated)
                