import sys
import numpy as np
from numpy import load
document = "PythonBCI\\data\\raw\\output_data_5.npz"
#document=input() 
np.set_printoptions(threshold=sys.maxsize)
data = load(document)
print(data)
lst = data.files
print(lst)
epoch=input()
print(data[lst[0]][int(epoch)][0])
#for item in lst:
#    print(item)
#    print(data[item])
