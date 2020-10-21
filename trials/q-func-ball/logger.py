from datetime import datetime
from pathlib import Path
import inspect
import pickle

class Logger(object):
    def __init__(self, agent):
        self.agent = agent
        self.reward_log = []
        self.step_count_log = []

    def reset(self):
        self.reward_log = []
        self.step_count_log = []

    def log(self, reward, step_count):
        self.reward_log.append(reward)
        self.step_count_log.append(step_count)

    def save_result(self, viewer):
        # 以下結果の表示
        timestamp = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
        log_dir = Path(__file__).parent / f'logs/{timestamp}'
        log_dir.mkdir(exist_ok=True)

        # 報酬関数のコードを保存
        with open(log_dir / 'func.txt', 'w') as f:
            f.write(
                inspect.getsource(viewer.env.reward_func)
            )

        # ログをpickleデータとして保存
        with open(log_dir / 'q.pickle', 'wb') as f:
            pickle.dump(dict(self.agent.Q), f)

        # グラフを画像で保存
        viewer.plot_result(str(log_dir / 'fig.png'))
