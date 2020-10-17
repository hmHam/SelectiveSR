'''
ボールの座標xを
[+1, x2, /2, -1, end]のactionを
用いてより早く100に近づける
Stateはボールの座標x
'''
# NOTE: 強化学習のベースを学習して -> タスクに特化する部分を考える

import argparse

from agent import BallAgent
from env import Env
from viewer import Viewer
from tester import Tester
from trainer import Trainer

# コマンドラインオプション
parser = argparse.ArgumentParser()
parser.add_argument('--episode_count', '-e', type=int, default=10000)
parser.add_argument('--interval', '-i', type=int, default=50)
parser.add_argument('--verbose', '-v', action='store_false')
parser.add_argument('--epsilon', '-ep', type=float, default=0.2)
args = parser.parse_args()

# 学習
agent = BallAgent(epsilon=args.epsilon)
env = Env()
trainer = Trainer(env, agent)
trainer.train(
    episode_count=args.episode_count,
    report_interval=args.interval,
    verbose=args.verbose
)

# テストフェーズ
tester = Tester(env, agent)
result = tester.test()
print(result)

# グラフを表示
viewer = Viewer(agent)
agent.logger.save_result(env, viewer)
viewer.plot_result(env, interval=args.interval)

