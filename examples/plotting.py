import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

f = open('../data/loudness/loudness_small_simple_test3_3.p', 'rb')
loudness = pickle.load(f)
f.close()

# f = open('output.p', 'rb')
# all_output = pickle.load(f)
# f.close()

# if "_pickle.UnpicklingError: the STRING opcode argument must be quoted" error,
# convert outputs pickle file to unix file endings using dos2unix.py in data folder
f = open('../data/abaqus_outputs/outputs_small_simple_test.p', 'rb')  #
data = pickle.load(f, encoding='latin1')
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_test.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()

displacements = {}
temperatures = {}
steps = ['Step-2', 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])


label2='Cooling'
label3='Heating'
'''        
plt.figure()
plt.plot(data['Time']['Step-2'],
         loudness['Step-2'], 'b', label='Cooling')
plt.plot(1.0 + np.array(data['Time']['Step-3']),
         loudness['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()

plt.figure()
plt.plot(data['Time']['Step-2'],
         displacements['Step-2'], 'b', label='Cooling')
plt.plot(1.0 + np.array(data['Time']['Step-3']),
         displacements['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()

plt.figure()
plt.plot(data['Time']['Step-2'],
         temperatures['Step-2'], 'b', label='Cooling')
plt.plot(1.0 + np.array(data['Time']['Step-3']),
         temperatures['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()
'''
plt.figure()
plt.plot(temperatures['Step-2'],
         displacements['Step-2'], 'b', label='Cooling')
plt.plot(temperatures['Step-3'],
         displacements['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()


plt.figure()
plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], 'b', label='Cooling')
'''
plt.plot(temperatures['Step-3'][:len(loudness['Step-3'])],
         loudness['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()
'''
'''
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(temperatures['Step-2'],
         displacements['Step-2'], 'b--', label='Cooling')
ax1.plot(temperatures['Step-3'],
         displacements['Step-3'], 'b', label='Heating')
ax2.plot(temperatures['Step-2'],
         loudness['Step-2'], 'k--', label='Cooling')
ax2.plot(temperatures['Step-3'],
         loudness['Step-3'], 'k', label='Heating')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Displacements (m)', color='b')
ax2.set_ylabel('Loudness (PLdB)', color='k')
plt.show()
'''
# Reproducing Pictures from loudness to check calculations
with open('../data/images/3Dpicture_test3_3.p', 'rb') as fid:
    pic_data = pickle.load(fid)
    
x = pic_data['x']
y = pic_data['y']
z = pic_data['z']
X = pic_data['X']
Y = pic_data['Y']
Z = pic_data['Z']
A = pic_data['A']
y0_list = pic_data['y0_list']
output = pic_data['output']
nx = 50
ny = 20

plt.figure()
plt.plot(y0_list, A)
plt.ylabel('Area along Mach Cone')
plt.xlabel('Distance along aircraft')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(X, Y, Z, c='b')
x, y, z = output.reshape(nx*ny, 3).T
ax.scatter(x, y, z, c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
