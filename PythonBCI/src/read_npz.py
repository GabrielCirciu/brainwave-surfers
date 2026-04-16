import sys
import numpy as np
from numpy import load
document=input() 
np.set_printoptions(threshold=sys.maxsize)
data = load(document)
print(data)
lst = data.files
print(lst)
epoch=input()
print(data[lst[0]][int(epoch)])
print(data[lst[1]][int(epoch)])
print(data[lst[2]][int(epoch)])
#for item in lst:
#    print(item)
#    print(data[item])
