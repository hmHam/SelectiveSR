from enum import IntEnum, auto
from random import randint

from __const import BORDER, TARGET_NUM


class Action(IntEnum):
    PLUS_ONE = 0
    TIMES_TWO = 1
    MINUS_ONE = 2
    DIVIDE_TWO = 3
    END = 4
  

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
    def __init__(self):
        self.agent_state = State(randint(0, BORDER))
        self.done = False
        self.step_count = 0

    def reset(self):
        # REVIEW: おかしいかもしれん
        self.agent_state = State(randint(0, BORDER))
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
        # FIXME: (要)重み付け, １項目: ステップ数のペナルティ
        return - BORDER / 5 + (BORDER- abs(state.x - TARGET_NUM))

    def _move(self, state, action):
        s = state.clone()
        if state.x == TARGET_NUM:
            self.done = True
            return s
        self.step_count += 1
        if action == Action.PLUS_ONE:
            s.x += 1
        elif action == Action.TIMES_TWO:
            s.x *= 2
        elif action == Action.MINUS_ONE:
            s.x -= 1
        elif action == Action.DIVIDE_TWO:
            s.x //= 2
        elif action == Action.END:
            # 値は変更しない
            self.done = True
            self.step_count -= 1
        # s.xが範囲を超えていたら終了して値を更新しない
        if s.x > BORDER or s.x < 0:
            self.done = True
            s = state.clone()
        return s

    def step(self, action):
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
        next_state = self._move(state, action)
        reward = self.reward_func(state)
        self.agent_state = next_state
        return next_state, reward, self.done
