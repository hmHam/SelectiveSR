'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える

from importlib import import_module
from pathlib import Path
import pickle

import sys
sys.path.append(str(Path(__file__).absolute()))

import inspect
import numpy as np

from env import Env, Action
from viewer import Viewer
from tester import Tester
from helpers import get_parser

def is_concrete_class(x, suffix):
    return suffix in x.__name__ and x.__name__[:-len(suffix)] != '' 

# 学習方法を決める
parser = get_parser()
args = parser.parse_args()

T = args.target_number
B = args.border

# seedを固定
np.random.seed(args.seed)

# 環境の構築
env = Env(T, B)

if args.play:
    with open('agent.pickle', 'rb') as f:
        agent = pickle.load(f)
    step_counts = []
    for x in range(B + 1):
        result = agent.play(env, x)
        result['path'] = [[s.x, Action.labels[a], ns.x] for (s, a, r, ns) in result['path']]
        print('-' * 10)
        print(f'start x = {x}')
        print('\t', '[s, a, r, ns]')
        for path in result['path']:
            print('\t', path)
        print('total reward', result['total_reward'])
        print('step_count', result['step_count'])
        print()
        step_counts.append(
            (x, result['step_count'])
        )
    print('step_counts for each start point')
    tester = Tester(env, agent, T, B)
    # NOTE: 最後のActionはENDなどで１引く
    tester.test_step_count(est=[e[1] - 1 for e in step_counts])
    for k, v in tester.results.items():
        print(k, v)
    sys.exit(0)

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
    verbose=args.verbose,
    lr=args.learning_rate,
    gamma=args.gamma
)

# 本格テスト
tester = None
if args.test:
    tester = Tester(env, agent, T, B)
    print('Agentの学習結果を評価します')
    tester.test_act()
    result = tester.test_step_count()


# 結果の可視化
viewer = Viewer(env, agent, args.interval, TARGET_NUM=T)
print('結果を表示します')
viewer.plot_result()

# ログ取り
print('ロギングします')
agent.logger.save_result(env, trainer, viewer, args, tester=tester)

print('Done')

# agentを保存
with open('agent.pickle', 'wb') as f:
    agent._Q = dict(agent._Q)
    pickle.dump(agent, f)