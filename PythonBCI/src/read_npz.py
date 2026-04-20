import sys
import numpy as np
from numpy import load

np.set_printoptions(threshold=sys.maxsize)

the_file = input("Enter the file name: ")
document = "PythonBCI\\data\\raw\\" + the_file
data = load(document)
lst = data.files

print('\nData: ', data)
print('Data keys: ', lst)

print(f'\Epoch 0 data')

print('\nEEG Data')
print('Shape: ', data[lst[0]].shape)
print('Signal 0 data: ', data[lst[0]][0][0])

print('\nAUX Data')
print('Shape: ', data[lst[1]].shape)
print('Signal 0 data: ', data[lst[1]][0][0])

print('\nLabels Data')
print('Shape: ', data[lst[2]].shape)
print('Trial 0 label: ', data[lst[2]][0])

print(f'\n')
