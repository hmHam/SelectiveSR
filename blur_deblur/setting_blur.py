from .blur_funcs import get_funcs
import numpy as np

CHANNEL = 1
WEIGHT = 0.0
OUTDIR = "results/ch1w000"
FILTERS = [lambda x: x] + [get_funcs(*delta) for delta in [
    (np.diag([10**2, 10**2]), 9),
    (np.diag([10**2, 1]), 9),
    (np.diag([1, 10**2]), 9),
    
]]

filt.append()
filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))
filt.append(lambda x: np.maximum(0, restoration.wiener(x, kernel1, 1e-2)))