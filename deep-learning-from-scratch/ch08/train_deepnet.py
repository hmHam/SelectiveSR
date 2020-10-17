# coding: utf-8
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from common.trainer import Trainer
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
