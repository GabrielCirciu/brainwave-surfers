import sys
import numpy as np
from numpy import load
document=input() 
np.set_printoptions(threshold=sys.maxsize)
data = load('output_data_'+document+'.npz')
lst = data.files
epoch=input()
print(data[lst[0]][0][int(epoch)])
#for item in lst:
#    print(item)
#    print(data[item])
