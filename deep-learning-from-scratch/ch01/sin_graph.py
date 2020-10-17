# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

# データの作成
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# グラフの描画
plt.plot(x, y)
plt.show()
