
from random import randint
import numpy as np
from .action import Action
from .state import State
D = 5


class Env(object):
    '''報酬を返す環境'''

    def __init__(self, TARGET_NUM, BORDER):
        self.border = BORDER
        self.target_num = TARGET_NUM
        self.state = State(randint(0, BORDER), BORDER)
        self.done = False
        self.step_count = 0

    def reset(self, x=None):
        if x is None:
            self.state = State(randint(0, self.border), self.border)
        else:
            self.state = State(x, self.border)
        self.done = False
        self.step_count = 0
        return self.state

    @property
    def actions(self):
        return [
            Action.PLUS_ONE,
            Action.TIMES_TWO,
            Action.END,
            Action.MINUS_ONE,
            Action.DIVIDE_TWO,
        ]

    def reward_func(self, state, n_state):
        total = 0
        if n_state.x > self.border or n_state.x < 0:
            # 範囲を超えたら超えた分だけペナルティ
            total += -abs(n_state.x - self.target_num)/self.border
        # TODO: TARGET_NUMから離れる方向の状態遷移はペナルティ
        diff = abs(self.target_num - n_state.x) - abs(self.target_num - state.x)
        if diff > 0:
            total -= diff / self.border
        else:
            total += diff / self.border
        # FIXME: ENDが全体的に評価高い -> 諦めずに頑張った方がいいと思ってほしい
        if self.done and n_state.x != self.target_num:
            return -3
        elif self.done:
            loss_reward = (self.border - abs(n_state.x - self.target_num))/self.border
            return loss_reward
        return -0.05 + total

    def _move(self, state, action):
        ns = state.clone()
        self.step_count += 1
        if action == Action.PLUS_ONE:
            Action.plus_one_action(ns)
        elif action == Action.TIMES_TWO:
            Action.times_two_action(ns)
        elif action == Action.MINUS_ONE:
            Action.minus_one_action(ns)
        elif action == Action.DIVIDE_TWO:
            Action.divide_two_action(ns)
        elif action == Action.END:
            # 値は変更しない
            Action.end_action(self)
            self.step_count -= 1
        if ns.x == self.target_num:
            self.done = True
        # 範囲外に出たら終了
        if ns.x > self.border or ns.x < 0:
            self.done = True
        return ns

    def step(self, action):
        state = self.state
        # 100かAction.ENDをとる場合が終了条件
        next_state = self._move(state, action)
        reward = self.reward_func(state, next_state)
        self.state = next_state
        return next_state, reward, self.done
