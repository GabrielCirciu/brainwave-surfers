import sys
import numpy as np
from numpy import load
np.set_printoptions(threshold=sys.maxsize)
data = load('output_data.npz')
lst = data.files
print(data[lst[0]][0][0])
#for item in lst:
#    print(item)
#    print(data[item])
