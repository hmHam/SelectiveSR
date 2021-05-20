import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from env import State, Action

class Tester(object):
    '''学習した結果を評価する'''
    def __init__(self, env, agent, TARGET_NUM, BORDER):
        self.target_num = TARGET_NUM
        self.border = BORDER
        self.agent = agent
        self.env = env
        self.results = {}
        self.logger = agent.logger
        self.results = None

    def test_act(self):
        # 最適戦略を取れているかをチェック
        for x in range(self.env.B + 1):
            s = State(x, self.env.B)
            print('state = %2d, opt_action = %s \t' % (x, Action.labels[np.argmax(self.agent.Q(s))]), self.agent.Q(s))

    def test_step_count(self, est=None):
        # 乱数の生成とその乱数に対する正解データの作成
        y = self._get_true_step_counts()
        # 学習したAgentでの結果を取得
        if est is None:
            est = self._get_estimates(self.agent)
        if not isinstance(est, np.ndarray):
            est = np.array(est)
        # 評価
        self.results = {
            'true_step_counts': y.tolist(),
            'est_step_counts': est.tolist(),
            'n': self.env.B + 1,
            'is_perfect': bool((y == est).mean()),
            'fail_initial_state': np.arange(self.env.B + 1)[y != est].tolist()
        }
        return self.results

    def _get_true_step_counts(self):
        return np.array([self._calc_min_step(s) for s in range(self.env.B + 1)])

    def _get_estimates(self, agent):
        '''エージェントに生成した乱数から推論させてかかったステップを記録'''
        # NOTE: agentの行動選択はENDが入っているので1を引く
        return np.array([agent.play(self.env, s)['step_count'] - 1 for s in range(self.env.B + 1)])

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
