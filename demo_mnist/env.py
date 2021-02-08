from scipy.signal import fftconvolve
import torch as tc
from torch.nn import functional as F

device = tc.device('cuda:0')

class InnerState(object):
    def __init__(self, now_img, init_img):
        if not tc.is_tensor(now_img):
            now_img = tc.tensor(now_img).to(device)
        if not tc.is_tensor(init_img):
            init_img = tc.tensor(init_img).to(device)
        self.x = tc.stack([now_img, init_img])
    
    @property
    def now(self):
        return self.x[0]
    
    @property
    def init(self):
        return self.x[1]   
    
    
def check_step_T(env, T):
    return env._step_count == T


class ReconstructionEnv(object):
    def __init__(self, trainloader, testloader, OuterStateClass, actions, T, blur_func, train=True,  judge_done=check_step_T):
        self._state = None
        self.train = train
        self._step_count = 0
        self.OuterStateClass = OuterStateClass
        self.blur_func = blur_func
        self.judge_done = judge_done
        self.trainloader = trainloader
        self.testloader = testloader
        self.actions = actions
        self.T = T
    
    def reset(self):
        '''
        数字が0のデータからサンプルを選んでくる。
        1)targetとなる画像を、self._bに
        2)targetをぼかした画像を、self._stateにセット
        '''
        self._step_count = 0
        
        #1) 目標画像の準備
        if self.train:
            self._b = next(iter(self.trainloader))[0]
        else:
            self._b = next(iter(self.testloader))[0]
        self._b = self._b.squeeze(0).squeeze(0).to(device)
        #2) ぼかした画像と初期の画像を2チャンネルで持った画像にしてみる
        self.init_state, self.decay_kernel_index = self.blur_func(self._b, self.T)
        self.init_state = tc.tensor(self.init_state).to(device)
        self._state = InnerState(self.init_state, self.init_state)
        
    @property
    def state(self):
        return self.OuterStateClass(self._state)
    
    def next_state(self, action):
        restore = tc.tensor(
            action(self._state.now.cpu().numpy())
        ).to(device, dtype=tc.float)
        restore = F.relu(restore)
        return InnerState(restore, self.init_state)
    
    def reward_func(self, state, next_state):
        # TODO: 報酬関数を以下のように設計
        return -tc.mean((self._b - next_state.now)**2)
    
    def step(self, a):
        self._step_count += 1
        action = self.actions[a]
        next_state = self.next_state(action)
        reward = self.reward_func(self._state, next_state)
        done = self.judge_done(self, self.T)
        self._state = next_state  # そのまま代入して大丈夫。今回の環境では次の状態が常に存在するので。
        return self.OuterStateClass(next_state), reward, done