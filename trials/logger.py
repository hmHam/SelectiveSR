from datetime import datetime
from pathlib import Path
import inspect
import json
from env import State

class Logger(object):
    def __init__(self, agent):
        self.agent = agent
        self.reward_log = []
        self.step_count_log = []
        timestamp = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
        self.log_dir = Path(__file__).parent / 'logs' / timestamp
        self.log_dir.mkdir(exist_ok=True)

    def reset(self):
        self.reward_log = []
        self.step_count_log = []

    def log(self, reward, step_count):
        self.reward_log.append(reward)
        self.step_count_log.append(step_count)

    def save_test_result(self, test_reults):
        with open(self.log_dir / 'test_result.json', 'w') as f:
            json.dump({
                'mean': test_reults.mean(),
                'size': test_reults.shape[0]
            }, f)

    def save_result(self, env):
        # 報酬関数のコードを保存
        with open(self.log_dir / 'func.txt', 'w') as f:
            f.write(
                inspect.getsource(env.reward_func)
            )

        # Q値のログを保存
        q_table = [list(self.agent.Q(State(x, env.B))) for x in range(env.B + 1)]
        with open(self.log_dir / 'q_table.json', 'w') as f:
            json.dump(q_table, f)

    def save_figure(self, viewer):
        # グラフを画像で保存
        viewer.plot_result(str(self.log_dir))
