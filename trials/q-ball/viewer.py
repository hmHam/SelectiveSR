import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.style.use('ggplot')



class Viewer(object):
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.logger

    def plot_result(self, env, log_path='', interval=500):
        # レイアウトの作成
        fig = plt.figure(figsize=(14, 6))
        plt.subplots_adjust(hspace=0.6)
        gs_master = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
        gs_1_and_2 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[:, 0])
        ax1 = fig.add_subplot(gs_1_and_2[0, :])
        ax2 = fig.add_subplot(gs_1_and_2[1, :])

        gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[:, 1])
        ax3 = fig.add_subplot(gs_3[:, :])
        
        # 学習段階での各エピソードの最終ステップの即時報酬をプロット
        self._plot_last_reward(ax1, interval)
        # ステップカウントの推移をプロット
        self._plot_step_count(ax2, interval)
        # 学習したQ-tableの値を表示
        self._plot_q_value(env, ax3)

        if log_path:
            fig.savefig(log_path)
        else:
            plt.show()
        
    def _plot_step_count(self, ax, interval):
        '''インターバル事のエピソードの平均ステップ数を表示'''
        indices = list(range(0, len(self.logger.step_count_log), interval))
        step_counts = np.array([self.logger.step_count_log[i:i+interval] for i in indices])
        means = step_counts.mean(axis=1)
        stds = step_counts.std(axis=1)

        ax.set_title('Step Count History')
        ax.grid()

        ax.fill_between(indices, means - stds, means + stds,
                            alpha=0.1, color="b")
        ax.plot(indices, means, "o-", color="b",
                    label=f"Step Counts for each {interval} episode")

    def _plot_last_reward(self, ax, interval):
        '''学習段階での各エピソードの最終ステップの即時報酬をプロット'''
        # プロットでも再利用するので先に変数化
        indices = list(range(0, len(self.logger.reward_log), interval))
        rewards = np.array([self.logger.reward_log[i:i+interval] for i in indices])
        means = rewards.mean(axis=1)
        stds = rewards.std(axis=1)

        ax.set_title("Reward History")
        ax.grid()
        ax.fill_between(indices, means - stds, means + stds,
                            alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g",
                    label=f"Rewards for each {interval} episode")
        ax.legend(loc="best")
        
    def _plot_q_value(self, env, ax):
        '''学習したQ-tableの値を表示'''
        vmin, vmax = min(self.agent.Q), max(self.agent.Q)
        kk = []
        for vals in self.agent.Q.values():
            kk.extend(vals)
        qmin, qmax = min(kk), max(kk)
        state_num = (vmax - vmin + 1)
        size = state_num * 3
        reward_map = np.zeros((3, size))
        for s, vals in self.agent.Q.items():
            _c = 3 * s + 1
            reward_map[1][_c] = 255 * vals[4] / (qmax - qmin) # END
            # NOTE: 反時計周りに正方向に大きく移動する順に並べた
            reward_map[1][_c + 1] = 255 * vals[1] / (qmax - qmin)# x2
            reward_map[0][_c] = 255 * vals[0]/ (qmax - qmin) # +1
            reward_map[1][_c - 1] = 255 * vals[2] / (qmax - qmin) # -1
            reward_map[2][_c] = 255 * vals[3] / (qmax - qmin) # /2
        ax.axvline(x=3 * 100 + 1, ymin=qmin, ymax=qmax)
        ax.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
        ax.set_xticks(np.arange(0, size, 3))
        ax.set_yticks([2])
        ax.set_xticklabels(np.arange(0, state_num))
        ax.set_yticklabels(np.arange(0, 1))
