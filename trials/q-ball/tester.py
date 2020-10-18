import numpy as np
from __const import BORDER, TARGET_NUM


class Tester(object):
    def __init__(self, env, agent):
        self.agent = agent
        self.env = env

    def test(self, n=100, seed=0):
        '''結果の表示'''
        # 乱数の生成とその乱数に対する正解データの作成
        np.random.seed(seed)
        start_points = np.random.randint(0, BORDER, n)
        y = np.array(self._get_labels(start_points))
        # 学習したAgentでの結果を取得
        est = np.array(self._get_estimates(self.agent, start_points))
        # 評価
        return {
            'size': n,
            'success': (y == est).sum()
        }

    def _get_labels(self, start_points):
        start_points = list(start_points)
        return [self._calc_min_step(s) for s in start_points]            

    @staticmethod
    def _calc_min_step(x):
        '''入力されたxに対する最小のステップ数を返す'''
        T = TARGET_NUM
        if x < T:
            if x == 0:
                # 全て+1
                return T
            # Tを超えずにxに２をかけられる最大の自然数t
            t = int(np.log2(T / x))
            # Tを一回だけ跨いだ数とその手前で一騎討ち
            return min([
                (T - x * pow(2, t)) + t,
                (x * pow(2, t + 1) - T) + (t + 1)
            ])
        elif x > T:
            # Tを下回らずにxを2で割れる最大の自然数t
            t = int(np.log2(x / T))
            # Tを一回だけ跨いだ数とその手前で一騎討ち
            return min([
                (x // pow(2, t) - T) + t,
                (T - x // pow(2, t + 1)) + (t + 1)
            ])
        return 0

    def _get_estimates(self, agent, start_points):
        '''エージェントに生成した乱数から推論させてかかったステップを記録'''
        return [agent.play(self.env, s)['step_count'] for s in start_points]
