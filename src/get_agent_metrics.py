### 学習済みのagentに対してテストデータに対する復元結果のMSEを評価する。
import torch
import numpy as np
from tqdm.notebook import tqdm

class Agent(object):
    def __init__(self, Qnet, action_candidates):
        self.Qnet = Qnet
        self.actions = [lambda x: x] + action_candidates
        
    def __call__(self, y, gpu):
        # 入力した画像をなおした結果と選んだ行動を返してほしい
        y = y.to(gpu, torch.float)
        took_actions = []
        for _ in range(5):
            with torch.no_grad():
                q, _ = self.Qnet(y[None, None, ...])
                q = q[0]
                _, a = torch.max(q, axis=0)
                took_actions.append(a.item())
            y = self.actions[a](y)
        return y, took_actions
    
    
def agent_metrics(test_dataset, Qnet, action_candidates, outdir=None, N=10000, gpu='cuda:0'):
    agent = Agent(Qnet, action_candidates)
    
    samples = []
    for test_img in tqdm(test_dataset[:N]):
        agent_output, agent_took_actions = agent(test_img, gpu)
        samples.append(
            agent_took_actions + [torch.mean((agent_output-test_img)**2).item()]
        )
    metrics = np.array(samples)
    if outdir is not None:
        np.save(metrics)
    return metrics