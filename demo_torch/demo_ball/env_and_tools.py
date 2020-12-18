import torch as tc
act = ['*2', '+1', 'end', '-1', '/2']
TIMES_TWO = 0
PLUS_ONE = 1
END = 2
MINUS_ONE = 3
DIVIDE_TWO = 4

SPAN_LENGTH = 11
MAX_VAL = 10
ACTION_DIM = 5
device = tc.device('cuda:0')

def x2vec(x):
    if not tc.is_tensor(x):
         x = tc.tensor(x).to(device)
    return F.one_hot(x, num_classes=SPAN_LENGTH).type(tc.float)


class InnerState(object):
    '''環境の内部でのState'''
    def __init__(self, x):
        self.x = x
        
# Linear実験用
# class OuterState(object):
#     def __init__(self, inner_state, init):
#         self.v = x2vec(inner_state.x)


# def f(x):
#     return 5
# end


# NN実験用
class OuterState(object):
    def __init__(self, inner_state, init):
#         self.v = tc.cat([x2vec(inner_state.x), x2vec(init)])
        self.v = tc.cat([x2vec(inner_state.x), (init/10.0).unsqueeze(0)])
        
        
def f(x):
    return 10 - x
# end 


class Env(object):    
    def __init__(self, x0):
        if not tc.is_tensor(x0):
            x0 = tc.tensor(x0).to(device)
        self.x0 = x0
        self.T = f(x0)
        self._state = InnerState(x0)
    
    def reset(self):
        self._state = InnerState(x0)
    
    @property
    def state(self):
        # Agent側に伝えるState
        return OuterState(self._state, self.x0)
    
    def next_state(self, s, a):
        '''see above action list for indecies'''
        if a == TIMES_TWO:
            x_next = s.x * 2
        elif a == PLUS_ONE:
            x_next = s.x + 1
        elif a == END:
#             x_next = s.x if s.x == self.T else self.x0 
            x_next = s.x
        elif a == MINUS_ONE:
            x_next = s.x - 1
        elif a == DIVIDE_TWO:
            x_next = s.x // 2
        return x_next
    
    def reward_func(self, state, action, x_next, done, penalty=-100.0):
        r = -1.0
        if action == END:
            if x_next != self.T:
                return penalty
            return (-1)*penalty
        else:
            # END以外で動いていない
            r = r + penalty if x_next == state.x else r
        diff = abs(x_next - self.T) - abs(state.x - self.T)
        if diff > 0:
            r += penalty
        if x_next < 0 or x_next > 10:
            r = r + penalty
        return r
    
    def step(self, a):
        x_next = self.next_state(self._state, a)
        # done = (a == END and ns.x == self.T)
        done = (a == END)
        r = self.reward_func(self._state, a, x_next, done)
        x_next = tc.max(tc.tensor(0).to(device), tc.min(tc.tensor(10).to(device), x_next))
        ns = InnerState(x_next)
        self._state = ns
        return OuterState(ns, self.x0), r, done
    
    
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import cm


class View:
    '''学習結果を可視化する'''
    @staticmethod
    def moving_average(r, n):
        ret = tc.cumsum(r, dim=0, dtype=tc.float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    @classmethod
    def show_moving_average(cls, R, n):
        '''報酬が学習につれてどう推移するかを可視化'''
        r = R.pop('all')
        plt.plot(cls.moving_average(r.cpu(), n))
        plt.xlabel('# of Trials')
        plt.ylabel('Avg. Reward')
        plt.show()
        
        plt.subplots_adjust(wspace=0.5)
        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        for init, r in R.items():
            ax = axes[init//6][(init%6)]
            ax.plot(cls.moving_average(r.cpu(), n))
            ax.set_xlabel('%d of Trials' % init)
            ax.set_ylabel('Avg. Reward')
        plt.show()

    @staticmethod
    def show_table(agent):
        plt.figure(figsize=(20, 18))
        plt.subplots_adjust(wspace=0.3)

        for init in range(SPAN_LENGTH):
            # TODO: 数字ごとに行動評価を可視化
            init = tc.tensor(init).to(device)
            T = f(init)
            U = tc.zeros((SPAN_LENGTH, agent.action_dim))
            for i in range(SPAN_LENGTH):
                i = tc.tensor(i).to(device)
                s = OuterState(InnerState(i), init).v
                with tc.no_grad():
                    U[i, :] = F.softmax(agent.Q_function(s), dim=0)
            init = init.cpu()
            U = U.cpu()
            plt.subplot(1, 11, int(init)+1)
            plt.title('init %d' % init)
            im = plt.imshow(U, cmap=cm.RdYlGn)
            plt.xticks(range(ACTION_DIM), act)
        plt.colorbar(im)
        plt.show()

    @classmethod
    def view(cls, agent, R, n=100):
        cls.show_moving_average(R, n=n)        
        cls.show_table(agent)
        
        
class Test:
    '''目標値にたどり着くかを確かめる'''
    @classmethod
    def test_all_init(cls, agent):
        '''[0, 10]の初期値から初めて、学習した結果で目標値にたどり着くかを検証'''
        solved = []
        for init in range(SPAN_LENGTH):
            init = tc.tensor(init).to(device)
            env = Env(init)
            s = env.state
            T = f(init)
            for _ in range(100):
                a = agent.policy(s, play=True)
                s, r, d = env.step(a)
                if d:
                    break
            # チート。本来強化学習の結果が「うまくいったか」の判断は、難しい。
            if env._state.x == T:
                solved.append((init, T, 'solved!'))
            else:
                solved.append((init, T, '-'))
        print('\n'.join('init = %d, T = %d, %s\n' % item for item in solved))

    @classmethod
    def test_specific_init(cls, agent, init, max_episode_length=10):
        '''指定した初期値から始まるエピソードで選択される行動系列と状態遷移を出力する'''
        init = tc.tensor(init).to(device)
        done = False
        env = Env(init)

        print('start from', int(env.x0))
        print('T =', f(env.x0))

        s = env.state

        visited_states = [int(env._state.x)]
        took_actions = []
        got_rewards = []
        for _ in range(max_episode_length):
            a = agent.policy(s, play=True)
            s, r, d = env.step(a)
            # log
            took_actions.append(act[a])
            # チート。本来OuterStateの画像しか知らない
            visited_states.append(int(env._state.x))
            got_rewards.append(float(r))
            if d:
                break
        print('visited states', visited_states)
        print('took actions', took_actions)
        print('got rewards', got_rewards)
        print('total reward = ', sum(got_rewards))