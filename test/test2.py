import numpy as np
import random
TARGET_NUM = 50
BORDER = 100

def calc_min_count(x):
    if x < TARGET_NUM:
        t = int(np.log2(TARGET_NUM / x))
        return min([
            (TARGET_NUM - x * pow(2, t)) + t,
            (x * pow(2, t + 1) - TARGET_NUM) + (t + 1)
        ])
    elif x > TARGET_NUM:
        t = int(np.log2(x / TARGET_NUM))
        return min([
            (x // pow(2, t) - TARGET_NUM) + t,
            (TARGET_NUM - x // pow(2, t + 1)) + (t + 1)
        ])
    return 0

if __name__ == '__main__':
    for d in [5, 13, 55, 80]:
        print('x = ', d)
        print(calc_min_count(d))
        print()