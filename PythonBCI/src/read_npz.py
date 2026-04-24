import sys
import numpy as np
import os
from numpy import load

np.set_printoptions(threshold=sys.maxsize)

the_file =  os.path.join("PythonBCI", "data", "raw", "hiamp-26-04-23-13-46", "batch_0.npz")

data = load(the_file)
lst = data.files

print('\nData: ', data)
print('Data keys: ', lst)

print('\nEEG Data')
print('Shape: ', data[lst[0]].shape)
print('Epoch 0 -> Signal 0 data: ', data[lst[0]][0][0])

print('\nAUX Data')
print('Shape: ', data[lst[1]].shape)
print('Epoch 0 -> Signal 0 data: ', data[lst[1]][0][0])

print('\nLabels Data')
print('Shape: ', data[lst[2]].shape)
print('Epoch 0 label: ', data[lst[2]][0])

print(f'\n')
