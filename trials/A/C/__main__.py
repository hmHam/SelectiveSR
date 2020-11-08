from glob import glob
import sys
del sys.path[0]
print('sys.path', sys.path)
print(glob('./*'))

from . import c
c.c()
