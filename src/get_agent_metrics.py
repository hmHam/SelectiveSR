### 学習済みのagentに対してテストデータに対する復元結果のMSEを評価する。
import torch
import numpy as np
from tqdm import tqdm

class Agent(object):
    def __init__(self, Qnet, actions, channel):
        self.Qnet = Qnet
        self.actions = actions
        self.channel = channel
        
    def __call__(self, y, T=16):
        y = y.to(torch.float)
        took_actions = []
        # 入力が1チェネルの場合
        if self.channel == 1:
            for _ in range(T):
                with torch.no_grad():
                    q, _ = self.Qnet(y[None, None, ...])
                    q = q[0]
                    _, a = torch.max(q, axis=0)
                    took_actions.append(a.item())
                y = self.actions[a](y)
            return y, took_actions
        # 2チャネルの場合
        s = torch.stack([y, y]).unsqueeze(0)
        for _ in range(T):
            with torch.no_grad():
                q, _ = self.Qnet(s)
                q = q[0]
                _, a = torch.max(q, axis=0)
                took_actions.append(a.item())
            z, y = s.squeeze(0)
            z = self.actions[a](z)
            s = torch.stack([z, y]).unsqueeze(0)
        return s[0][0], took_actions
    
    
def agent_metrics(Dy, Dx, Qnet, action_candidates, channel=1, outdir=None, N=10000):
    agent = Agent(Qnet, action_candidates, channel)
    
    metrics = []
    for i in tqdm(range(N)):
        test_img = Dy[i]
        x = Dx[i]
        
        agent_output, agent_took_actions = agent(test_img)  #復元
        metrics.append(
            agent_took_actions + [torch.mean((agent_output-x)**2).item()]
        )
    metrics = np.array(metrics)
    if outdir is not None:
        np.save(metrics)
    return metrics