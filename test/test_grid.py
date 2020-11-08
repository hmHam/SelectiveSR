import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.style.use('ggplot')


# レイアウトの作成
fig = plt.figure(figsize=(10, 8))
gs_master = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
gs_1_and_2 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[:, 0])
ax1 = fig.add_subplot(gs_1_and_2[0, :])
ax2 = fig.add_subplot(gs_1_and_2[1, :])

gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[:, 1])
ax3 = fig.add_subplot(gs_3[:, :])

# データの作成
x1 = np.arange(10000)
y1 = x1

x2 = np.arange(10000)
y2 = -x2

x3 = np.arange(10)
y3 = np.power(x3, 2)

# プロット
ax1.plot(x1, y1, color='g')
ax2.plot(x2, y2, color='b')
ax3.plot(x3, y3, color='r')
plt.show()
