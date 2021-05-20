from enum import IntEnum, auto

class Action:
    labels = ['+1', '*2', 'end', '-1', '/2']
    
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
    