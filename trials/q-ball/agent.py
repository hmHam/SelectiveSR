from collections import defaultdict
# trialsディレクトリ直下のagent.pyから親クラスをimport
from agent import Agent

class QBallAgent(Agent):
    def __init__(self, env, epsilon):
        super().__init__(env, epsilon)
        self._Q = defaultdict(lambda: [0] * env.action_num)

    def Q(self, s, a=None):
        '''行動評価関数'''
        if a is None:
            return self._Q[s]
        return self._Q[s][a]