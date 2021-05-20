from datetime import datetime
from pathlib import Path
import inspect
import json
import numpy as np
from env import State

class Logger(object):
    '''実験結果を保存するクラス'''
    def __init__(self, agent):
        self.agent = agent
        self.reward_log = []
        self.step_count_log = []
        timestamp = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
        self.timestamp = timestamp
        self.log_dir = Path(__file__).parent / 'logs' / agent.__class__.__name__
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def reset(self):
        self.reward_log = []
        self.step_count_log = []

    def log(self, reward, step_count):
        self.reward_log.append(reward)
        self.step_count_log.append(step_count)

    def _get_trial_params(self, args):
        '''実験条件のパラメータをjsonで保存'''
        return {
            'pkg': args.package,
            'T': args.target_number,
            'B': args.border,
            'seed': args.seed,
            'train_episode_count': args.episode_count,
            'epsilon': args.epsilon,
            'train_method': args.trainer,
            'gamma': args.gamma,
            'learning_rate': args.learning_rate,
        }
    
    def _save_reward_func(self, env):
        # 報酬関数のコードを保存
        with open(self.log_dir / 'func.py', 'w') as f:
            f.write(
                inspect.getsource(env.reward_func)
            )

    def _get_state_evals(self, env):
        return [list(self.agent.Q(State(x, env.B))) for x in range(env.B + 1)]

    def save_result(self, env, trainer, viewer, args, tester=None):
        self.log_dir = self.log_dir / trainer.__class__.__name__ / self.timestamp
        self.log_dir.mkdir(exist_ok=True, parents=True)

        context = {}
        context['trial_params'] = self._get_trial_params(args)
        context['q'] = self._get_state_evals(env)
        if tester is not None:
            context['test_result_num'] = tester.results

        with open(self.log_dir / 'result.json', 'w') as f:
            json.dump(context, f, indent=2)

        self._save_figure(viewer)

    def _save_figure(self, viewer):
        '''グラフを画像で保存'''
        viewer.plot_result(str(self.log_dir))
