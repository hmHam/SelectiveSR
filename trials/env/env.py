import numpy as np
from .action import Action
from .state import State

class Env(object):
    '''報酬を返す環境'''

    def __init__(self, TARGET_NUM, BORDER):
        self.B = BORDER
        self.T = TARGET_NUM
        self.agent_state = None
        self.done = False
        self.step_count = 0

    def reset(self, x=None):
        if x is None:
            self.agent_state = State(np.random.randint(self.B + 1), self.B)
        else:
            self.agent_state = State(x, self.B)
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

    @property
    def action_num(self):
        return len(self.actions)

    def reward_func(self, state, n_state):
        total = 0
        if n_state.x > self.B or n_state.x < 0:
            # 範囲を超えたら超えた分だけペナルティ
            total += -abs(n_state.x - self.T)/self.B
        # TODO: TARGET_NUMから離れる方向の状態遷移はペナルティ
        diff = abs(self.T - n_state.x) - abs(self.T - state.x)
        if diff > 0:
            total += -3 * diff
        # FIXME: ENDが全体的に評価高い -> 諦めずに頑張った方がいいと思ってほしい
        if self.done and n_state.x != self.T:
            return -2
        elif self.done:
            # step_reward = 0.55 * (abs(self.target_num - n_state.x) - self.step_count)/self.target_num
            loss_reward = 2 * (self.B - abs(n_state.x - self.T))/self.B
            return loss_reward
        return -0.5 + total

    def _move(self, state, action):
        ns = state.clone()
        self.step_count += 1
        if action == 0:
            ns.x = state.x + 1
        elif action == 1:
            ns.x = state.x * 2
        elif action == 2:
            # END
            self.step_count -= 1
        elif action == 3:
            ns.x = state.x - 1
        elif action == 4:
            ns.x = state.x // 2
        return ns

    def step(self, action):
        self.done = (action == 2)
        state = self.agent_state
        # 100かAction.ENDをとる場合が終了条件
        next_state = self._move(state, action)
        reward = self.reward_func(state, next_state)
        # next_stateを0 - 10に丸める
        # self.agent_state = State(max(0, min(self.border, next_state.x)))
        self.agent_state = next_state
        return self.agent_state, reward, self.done