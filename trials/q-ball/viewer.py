import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import japanize_matplotlib
plt.style.use('ggplot')


class Viewer(object):
    def __init__(self, env, agent, interval, TARGET_NUM):
        self.target_num = TARGET_NUM
        self.env = env
        self.agent = agent
        self.interval = interval
        self.logger = agent.logger

    def plot_result(self, log_path=''):
        # レイアウトの作成
        fig = plt.figure(figsize=(20, 8))
        plt.subplots_adjust(hspace=0.6)
        gs_master = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
        gs_1_and_2 = GridSpecFromSubplotSpec(
            nrows=2, ncols=1, subplot_spec=gs_master[:, 0])
        ax1 = fig.add_subplot(gs_1_and_2[0, :])
        ax2 = fig.add_subplot(gs_1_and_2[1, :])

        gs_3 = GridSpecFromSubplotSpec(
            nrows=2, ncols=1, subplot_spec=gs_master[:, 1])
        ax3 = fig.add_subplot(gs_3[:, :])

        # 学習段階での各エピソードの最終ステップの即時報酬をプロット
        self._plot_last_reward(ax1, self.interval)
        # ステップカウントの推移をプロット
        self._plot_step_count(ax2, self.interval)
        # 学習したQ-tableの値を表示
        self._plot_q_value(self.env, ax3, fig)

        if log_path:
            fig.savefig(log_path)
        else:
            plt.show()

    def _plot_last_reward(self, ax, interval):
        '''学習段階での各エピソードの最終ステップの即時報酬をプロット'''
        # プロットでも再利用するので先に変数化
        indices = list(range(0, len(self.logger.reward_log), interval))
        rewards = np.array([self.logger.reward_log[i:i+interval]
                            for i in indices])
        means = rewards.mean(axis=1)
        stds = rewards.std(axis=1)

        ax.set_title(f"最終ステップの即時報酬の平均推移/{interval}episode毎")
        ax.set_ylabel('最終ステップの即時報酬の平均')
        ax.set_xlabel('学習時のエピソードの番号')
        ax.grid()
        ax.fill_between(indices, means - stds, means + stds,
                        alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g",
                label="即時報酬")
        ax.legend(loc="best")

    def _plot_step_count(self, ax, interval):
        '''インターバル事のエピソードの平均ステップ数を表示'''
        indices = list(range(0, len(self.logger.step_count_log), interval))
        step_counts = np.array(
            [self.logger.step_count_log[i:i+interval] for i in indices])
        means = step_counts.mean(axis=1)
        stds = step_counts.std(axis=1)

        ax.set_title(f"ステップ数の平均/{interval}episode毎")
        ax.set_ylabel('ステップ数の平均')
        ax.set_xlabel('学習時のエピソードの番号')
        ax.grid()

        ax.fill_between(indices, means - stds, means + stds,
                        alpha=0.1, color="b")
        ax.plot(indices, means, "o-", color="b",
                label="ステップ数")
        ax.legend(loc='best')

    def _plot_q_value(self, env, ax, fig):
        '''学習したQ-tableの値を表示'''
        # TODO:
        #   * TARGET_NUMに水平線を引く
        #   * メモリをきちんとつける
        #   * set_xlabel, set_ylabelで状態, actionを示す
        ax.set_title('Value of Q-table')
        ax.axhline(y=self.target_num, xmin=0, xmax=len(env.actions) - 1)
        ax.set_xlabel('action')
        ax.set_ylabel('state')
        q_table = np.array([self.agent.Q[k] for k in sorted(self.agent.Q)])
        im = ax.imshow(q_table, cmap=cm.RdYlGn, aspect='auto',
                       vmax=abs(q_table).max(), vmin=-abs(q_table).max())
        fig.colorbar(im)
        ax.grid()
        ax.set_xticks(range(len(env.actions)))
        ax.set_xticklabels(['+1', 'x2', 'END', '-1', '/2'])
        ax.set_yticks(range(0, q_table.shape[0], q_table.shape[0]//10))
        ax.set_yticklabels(range(0, q_table.shape[0], q_table.shape[0]//10))
