### 学習済みのagentに対してテストデータに対する復元結果のMSEを評価する。
import torch
import numpy as np
from tqdm import tqdm

class Agent(object):
    def __init__(self, Qnet, actions, channel):
        self.Qnet = Qnet
        self.actions = actions
        self.channel = channel
        
    def __call__(self, y):
        # 入力した画像をなおした結果と選んだ行動を返してほしい
        y = y.to(torch.float)
        if self.channel == 1:
            feature = y[None, None, ...]
        elif self.channel == 2:
            feature = torch.stack([y, y]).unsqueeze(0)
        took_actions = []
        for _ in range(5):
            with torch.no_grad():
                q, _ = self.Qnet(feature)
                q = q[0]
                _, a = torch.max(q, axis=0)
                took_actions.append(a.item())
            if self.channel == 1:
                feature = self.actions[a](feature.squeeze(0).squeeze(0))
                feature = feature.unsqueeze(0).unsqueeze(0)
            else:
                y, z = feature.squeeze(0)
                z = self.actions[a](z)
                feature = torch.stack([y, z]).unsqueeze(0)
        return feature if self.channel == 1 else feature[0][1], took_actions
    
    
def agent_metrics(test_dataset, origin, Qnet, action_candidates, channel=1, outdir=None, N=10000):
    agent = Agent(Qnet, action_candidates, channel)
    
    samples = []
    for i in tqdm(range(N)):
        test_img = test_dataset[i]
        o = origin[i]
        
        agent_output, agent_took_actions = agent(test_img)
        samples.append(
            agent_took_actions + [torch.mean((agent_output-o)**2).item()]
        )
    metrics = np.array(samples)
    if outdir is not None:
        np.save(metrics)
    return metrics