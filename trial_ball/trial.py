'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える

from importlib import import_module
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).absolute()))

import inspect
import numpy as np

from env import Env, State
from viewer import Viewer
from tester import Tester
from helpers import get_parser


# 学習方法を決める
parser = get_parser()
args = parser.parse_args()

T = args.target_number
B = args.border

# seedを固定
np.random.seed(args.seed)

# 環境の構築
env = Env(T, B)

def is_concrete_class(x, suffix):
    return suffix in x.__name__ and x.__name__[:-len(suffix)] != '' 

# Agentの選択
AgentClass = inspect.getmembers(
    import_module('.agent', package=args.package),
    lambda x: inspect.isclass(x) and is_concrete_class(x, 'Agent')
)[0][1]
agent = AgentClass(env, epsilon=args.epsilon)

# 訓練用クラスの選択
TrainerClass = inspect.getmembers(
    import_module(f".trainers.{args.trainer}", package=args.package),
    lambda x: inspect.isclass(x) and is_concrete_class(x, 'Trainer')
)[0][1]
trainer = TrainerClass(env, agent, B)

print('Agentの学習を開始します')
trainer.train(
    episode_count=args.episode_count,
    report_interval=args.interval,
    verbose=args.verbose
    learning_rate=args.learning_rate,
    gamma=args.gamma
)

# 本格テスト
tester = None
if args.test:
    tester = Tester(env, agent, T, B)
    print('Agentの学習結果を評価します')
    result = tester.test(trial_count=args.test_count)

# 簡易テスト
act = ['+1', '-1', '*2', '/2', 'end']
for x in range(11):
    s = State(x, B)
    print('state = %2d, opt_action = %s \t' % (x, act[np.argmax(agent.Q(s))]), agent.Q(s))

# 結果の可視化
viewer = Viewer(env, agent, args.interval, TARGET_NUM=T)
print('結果を表示します')
viewer.plot_result()

# ログ取り
print('ロギングします')
agent.logger.save_result(env, trainer, viewer, args, tester=tester)

print('Done')