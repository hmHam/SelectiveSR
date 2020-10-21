from enum import IntEnum, auto
from random import randint
from scipy.special import eval_legendre
import numpy as np
D = 5


class Action(IntEnum):
    PLUS_ONE = 0
    TIMES_TWO = 1
    END = 2
    MINUS_ONE = 3
    DIVIDE_TWO = 4

    @classmethod
    def plus_one_action(cls, state):
        state.x += 1

    @classmethod
    def times_two_action(cls, state):
        state.x *= 2

    @classmethod
    def minus_one_action(cls, state):
        state.x -= 1

    @classmethod
    def divide_two_action(cls, state):
        state.x //= 2

    @classmethod
    def end_action(cls, env):
        env.done = True


class State(object):
    def __init__(self, x):
        # TODO: ルジャンドル多項式を用いて分散表現を作る
        vec = []
        for i in range(D):
            pass

    def clone(self):
        return State(self.x)


class Env(object):
    '''報酬を返す環境'''

    def __init__(self, TARGET_NUM, BORDER):
        self.border = BORDER
        self.target_num = TARGET_NUM
        self.agent_state = State(randint(0, BORDER))
        self.done = False
        self.step_count = 0

    def reset(self, x=None):
        if x is None:
            self.agent_state = State(randint(0, self.border))
        else:
            self.agent_state = State(x)
        self.done = False
        self.step_count = 0
        return self.agent_state

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
        if n_state.x > self.border or n_state.x < 0:
            # 範囲を超えたら超えた分だけペナルティ
            return -abs(n_state.x - self.target_num)/self.border
        # TODO: TARGET_NUMから離れる方向の状態遷移はペナルティ
        diff = abs(self.target_num - n_state.x) - abs(self.target_num - state.x)
        if diff > 0:
            return -3 * diff
        # FIXME: ENDが全体的に評価高い -> 諦めずに頑張った方がいいと思ってほしい
        if self.done:
            step_reward = 0.55 * (abs(self.target_num - n_state.x) - self.step_count)/self.target_num
            loss_reward = 2 * (self.border - abs(n_state.x - self.target_num))/self.border
            return step_reward + loss_reward
        return -0.5

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
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
        next_state = self._move(state, action)
        reward = self.reward_func(state, next_state)
        self.agent_state = next_state
        return next_state, reward, self.done
