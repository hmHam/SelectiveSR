from enum import IntEnum, auto
from random import randint


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
        env.step_count -= 1


class State(object):
    # TODO: Q学習がうまくいかない
    # solution:
    # (1) 偶数か奇数かなど xをいくつかの細かい情報にわける -> ベクトル化
    # (2) Q-tableをNNなどで近似 <- 分散表現が
    def __init__(self, x):
        self.x = x

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
            Action.MINUS_ONE,
            Action.DIVIDE_TWO,
            Action.END,
        ]

    def reward_func(self, state):
        if state.x > self.border or state.x < 0:
            # 範囲を超えたら超えた分だけペナルティ
            return - abs(state.x - self.target_num)
        return (- self.border / 5) + (self.border - abs(state.x - self.target_num))

    def _move(self, state, action):
        s = state.clone()
        if state.x == self.target_num:
            self.done = True
            return s
        self.step_count += 1
        if action == Action.PLUS_ONE:
            Action.plus_one_action(s)
        elif action == Action.TIMES_TWO:
            Action.times_two_action(s)
        elif action == Action.MINUS_ONE:
            Action.minus_one_action(s)
        elif action == Action.DIVIDE_TWO:
            Action.divide_two_action(s)
        elif action == Action.END:
            # 値は変更しない
            Action.end_action(self)
        # FIXME: 範囲外の状態も取れるようにする
        if s.x > self.border or s.x < 0:
            self.done = True
        return s

    def step(self, action):
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
        next_state = self._move(state, action)
        reward = self.reward_func(next_state)
        self.agent_state = next_state
        return next_state, reward, self.done
