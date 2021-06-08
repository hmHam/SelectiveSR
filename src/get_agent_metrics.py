### 学習済みのagentに対してテストデータに対する復元結果のMSEを評価する。
import os
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
        if self.channel == 1:
            z = y
            took_actions = []
            for _ in range(5):
                with torch.no_grad():
                    q, _ = self.Qnet(z[None, None, ...])
                    q = q[0]
                    _, a = torch.max(q, axis=0)
                    took_actions.append(a.item())
                z = z.cpu().numpy()
                z = self.actions[a](z)
                z = torch.from_numpy(z)
            return z, took_actions
        # 2チャンネル
        s = torch.stack([y, y]).unsqueeze(0)
        took_actions = []
        for _ in range(5):
            with torch.no_grad():
                q, _ = self.Qnet(s)
                q = q[0]
                _, a = torch.max(q, axis=0)
                took_actions.append(a.item())
            z, y = s.squeeze(0)
            z = z.cpu().numpy()
            z_next = self.actions[a](z)
            z_next = torch.from_numpy(z_next).to(torch.float).to(self.Qnet.scale.device)
            s = torch.stack([z_next, y]).unsqueeze(0)
        return s[0][0], took_actions
    
    
def agent_metrics(test_dataset, origin, Qnet, action_candidates, channel=1, outdir=None, N=10000):
    # deviceを統一
    device = Qnet.scale.device
    test_dataset = test_dataset.to(device).to(torch.float)
    origin = origin.to(device).to(torch.float)
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

def random_metrics(Dy, Dx, actions, seed):
    np.random.seed(seed)
    aN = np.random.choice(len(actions), (Dy.shape[0], 5))
    for n in range(Dy.shape[0]):
        xn = Dx[n]
        yn = Dy[n].copy()
        an = aN[n]
        # ステップ数
        for a in an:
            yn = actions[a](yn)
        random_mse.append(
            np.mean((yn - xn)**2)
        )
    return np.array(random_mse)


def model_metrics(Dy, Dx, actions, Qnet):
    pass

def get_metrics_col(data_context, actions, outdir, Qnets=[], seed=0):
    '''
    input:
        data_context(テストデータ),
        actions(Agentの行動の選択肢),
        outdir(結果の保存先),
        Qnets(list of 性能を測定したいQ関数のインスタス, length=I)
    return:
        col(初期画像, ランダム, Qnets[0], ..., Qnets[I]のMSEを測定した配列, length=I+2)
    '''
    Dx = data_context['original_dataset']
    Dy = data_context['test_dataset']

    # 入力画像のMSE
    init_mse = np.mean((Dy - Dx)**2, axis=(1, 2))
    # ランダムに行動を選択した場合のMSE
    random_mse = random_metrics(Dy, Dx, actions, seed)
    I = len(Qnets)
    Dy = torch.from_numpy(Dy)
    Dx = torch.from_numpy(Dx)
    Qnets_mse = []
    for i in range(I):
        Qnet = Qnets[i]
        channel = Qnet.channel
        results = agent_metrics(Dy.clone(), Dx, Qnet, actions, channel=channel)
        Qnets_mse.append(results[:, -1])
    # TODO: 全ての配列で数が一致することをassertion
    
    # 全ての1次元配列の結果を保存
    out = np.stack([init_mse, random_mse] + Qnets_mse, axis=1)
    np.save(os.path.join(outdir, 'metrics.npy'), out)
    return out.mean(axis=0)

def get_metrics_table(Datasets, actions, Qnets, outdir_base, seed=0):
    '''
    input: 
        Datasets(列で表示するデータセット),
        actions(行動の選択肢),
        outdir_base(実験名),
        Qnets(性能を測定したいモデルのリスト)
    return:
        table(行が手法, 列がデータセット, 要素がMSEの2次元配列)
    '''
    outdir_base = str(outdir_base)  #文字列であることを保証
    J = len(Datasets)
    # TODO:(INSIDE): J > 0をassert
    outdir = os.path.join(outdir_base, 'dataset0')
    table = get_metrics_col(Datasets[0])
    for j in range(1, J):
        context = Datasets[j]
        # FIXME(OUTSIDE): 
        #     data/ディレクトリ配下の全てのデータセットのキーを修正
        #     original_dataset -> original_imgs
        #     test_dataset, train_dataset -> blurred_imgs
        outdir = os.path.join(outdir_base, f'dataset{j}')
        col = get_metrics_col(context, actions, outdir, Qnets)
        table = np.c_[table, col]
    # 結果を保存
    np.save(os.path.join(outdir_base, 'metrics_table.npy'), table)
    return table