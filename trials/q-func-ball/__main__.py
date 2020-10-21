'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える

import argparse

import numpy as np

from agent import BallAgent
from env import Env
from viewer import Viewer
from tester import Tester
from trainer import Trainer

# コマンドラインオプション
parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--episode_count_ratio', '-e', type=int, default=1)
parser.add_argument('--interval', '-i', type=int, default=500)
parser.add_argument('--verbose', '-v', action='store_false')
parser.add_argument('--epsilon', '-ep', type=float, default=0.2)
# 目標の点と上界はここで決める
parser.add_argument('--target-number', '-t', type=int, default=5)
parser.add_argument('--border', '-b', type=int, default=10)
args = parser.parse_args()

# seedを固定
np.random.seed(args.seed)

# 学習
print('Agentの学習を開始します')
agent = BallAgent(epsilon=args.epsilon)
env = Env(args.target_number, args.border)
trainer = Trainer(env, agent, args.border)
trainer.train(
    episode_count=args.episode_count_ratio * 10000,
    report_interval=args.interval,
    verbose=args.verbose
)

# テスト
print('Agentの学習結果を評価します')
tester = Tester(
    env,
    agent,
    args.target_number,
    args.border
)
result = tester.test()
print(result)

# グラフを表示
viewer = Viewer(
    env,
    agent,
    interval=args.interval,
    TARGET_NUM=args.target_number
)
viewer.plot_result()
agent.logger.save_result(viewer)
