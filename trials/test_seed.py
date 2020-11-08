import numpy as np
from test_seed_out import Hoge

N = 1000000
np.random.seed(0)

a = np.random.randint(0, 11, 10)
h = Hoge()
res = []
for i in range(N):
    res.append(h.random_sample(1)[0])
a = np.r_[a, res]


np.random.seed(0)
b = np.random.randint(0, 11, N + 10)

print(all(a == b))
