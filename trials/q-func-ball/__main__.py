'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える

import argparse

import numpy as np
import random

from agent import BallAgent
from env import Env
from viewer import Viewer
from tester import Tester
from trainer import Trainer

# コマンドラインオプションを作成
parser = argparse.ArgumentParser()
parser.add_argument('--play', '-p', action='store_true', help='学習したモデルを使用')
parser.add_argument('--test', '-tst', action='store_true', help='学習したモデルが目標値にたどり着くまでの回数をテスト')

parser.add_argument('--seed', '-s', type=int, default=0, help='数値実験全体の乱数の種を指定')
EPISODE_BASE_COUNT = 10000
parser.add_argument('--episode_count_ratio', '-e', type=int, default=1, help=f'学習するエピソード数 = -ep * {EPISODE_BASE_COUNT}')
parser.add_argument('--interval', '-i', type=int, default=50, help='学習進捗を表示するインターバル')
parser.add_argument('--verbose', '-v', action='store_false', help='学習進捗を表示するかどうか')
parser.add_argument('--epsilon', '-ep', type=float, default=0.2, help='ε-Greedy法のパラメータepsilon')

# 目標の点と上界はここで決める
parser.add_argument('--target-number', '-t', type=int, default=5, help='目標の値')
parser.add_argument('--border', '-b', type=int, default=10, help='状態が取れる最大値')

args = parser.parse_args()

# seedを固定
# NOTE: ここで固定するの機能分離できてない
np.random.seed(args.seed)
random.seed(args.seed)
env = Env(args.target_number, args.border)

# 学習
print('Agentの学習を開始します')
agent = BallAgent(args.epsilon, env.state.size, env.actions)
trainer = Trainer(env, agent, args.border)
trainer.train(
    episode_count=args.episode_count_ratio * EPISODE_BASE_COUNT,
    report_interval=args.interval,
    verbose=args.verbose
)

# テスト
tester = Tester(
    env,
    agent,
    args.target_number,
    args.border
)
if args.test:
    print('Agentの学習結果を評価します')
    result = tester.test()
    print(result)

# グラフを表示
print('学習結果のグラフを表示します')
viewer = Viewer(
    env,
    agent,
    tester,
    interval=args.interval,
    TARGET_NUM=args.target_number,
    BORDER=args.border
)
viewer.plot_result(args.test)
agent.logger.save_figure(viewer)
