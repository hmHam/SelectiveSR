import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


class Tester(object):
    '''学習した結果を評価する'''
    def __init__(self, env, agent, TARGET_NUM, BORDER):
        self.target_num = TARGET_NUM
        self.border = BORDER
        self.agent = agent
        self.env = env
        self.logger = agent.logger
        self.results = None

    def test(self, n=100, seed=0, trial_count=1, visual=True):
        '''結果を配列で返す'''
        self.results = np.array([self.test_once(n, seed + t)['success'] for t in range(trial_count)])
        self.logger.save_test_result(self.results)
        return {
            'size': n,
            'results': self.results,
        }

    def test_once(self, n, seed):
        # 乱数の生成とその乱数に対する正解データの作成
        np.random.seed(seed)
        start_points = np.random.randint(0, self.border, n)
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

    def _calc_min_step(self, x):
        '''入力されたxに対する最小のステップ数を返す'''
        if x == 0:
            return 1 + self._calc_min_step(1)
        T = self.target_num
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
