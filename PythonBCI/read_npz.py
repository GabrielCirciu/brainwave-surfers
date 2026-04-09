from numpy import load

data = load('calib_data.npz')
lst = data.files
print(data[lst[0]][0][0])
#for item in lst:
#    print(item)
#    print(data[item])
