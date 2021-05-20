'''
* MNIST画像に対してガウスフィルターを畳み込みんで、表示
* 逆変換をして表示
'''

import numpy as np
from PIL import Image
from scipy.signal import convolve

img = Image.open('lena.png')
img_array = np.asarray(img).transpose(2, 0, 1)

# 3x3のガウシアンフィルタ
G_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16
out_by3x3 = np.zeros(img_array.shape)
for i in range(out_by3x3.shape[0]):
    out_by3x3[i] = convolve(img_array[i], G_3x3, mode='same')

out_img_by3x3 = Image.fromarray(np.uint8(out_by3x3.transpose(1, 2, 0)))
out_img_by3x3.save('blured_lena_3x3.png')


# 5x5のガウシアンフィルタs
G_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256

out_by5x5 = np.zeros(img_array.shape)
for i in range(out_by5x5.shape[0]):
    out_by5x5[i] = convolve(img_array[i], G_5x5, mode='same')

out_img_by5x5 = Image.fromarray(np.uint8(out_by5x5.transpose(1, 2, 0)))
out_img_by5x5.save('blured_lena_5x5.png')