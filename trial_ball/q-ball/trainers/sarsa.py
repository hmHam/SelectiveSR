from .td import TDTrainer

class SARSATrainer(TDTrainer):
    def update(self, agent, s, a, reward, s_next, gamma, lr):
        a_next = agent.policy(s_next)
        gain = reward + gamma * agent._Q[s_next][a_next]
        estimated = agent._Q[s][a]
        agent._Q[s][a] = agent._Q[s][a] + lr * (gain - estimated)
                