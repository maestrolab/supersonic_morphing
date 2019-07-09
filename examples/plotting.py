import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import pickle

<<<<<<< HEAD
f = open('../data/loudness/loudness_small_simple_test6_1.p', 'rb')
=======
f = open('../data/loudness/loudness_small_simple_test5_5_ux.p', 'rb')
>>>>>>> 447ba5deb91b40d9caa97b456342634059510125
loudness = pickle.load(f)
f.close()

# f = open('output.p', 'rb')
# all_output = pickle.load(f)
# f.close()

# if "_pickle.UnpicklingError: the STRING opcode argument must be quoted" error,
# convert outputs pickle file to unix file endings using dos2unix.py in data folder
f = open('../data/abaqus_outputs/outputs_small_simple_test_ux.p', 'rb')  #
data = pickle.load(f, encoding='latin1')
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_test_ux.p', 'rb')  #
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
<<<<<<< HEAD
with open('../data/images/3Dpicture_test6_1.p', 'rb') as fid:
=======
with open('../data/images/3Dpicture_test5_5_ux.p', 'rb') as fid:
>>>>>>> 447ba5deb91b40d9caa97b456342634059510125
    pic_data = pickle.load(fid)
# points from Mach cone intersections
x = pic_data['x']
y = pic_data['y']
z = pic_data['z']
# points from Abaqus ( three times)
X = pic_data['X']
Y = pic_data['Y']
Z = pic_data['Z']
A = pic_data['A']
# displacement from Abaqus (one increment)
U1 = pic_data['U1']
U2 = pic_data['U2']
U3 = pic_data['U3']
y0_list = pic_data['y0_list']
output = pic_data['output']
# Initial points from Abaqus (only one)
xo = pic_data['xo']
yo = pic_data['yo']
zo = pic_data['zo']
# mesh lengths
nx = 50
ny = 20

plt.figure()
plt.plot(y0_list, A)
plt.ylabel('Area along Mach Cone')
plt.xlabel('Distance along aircraft')

'''
with open('../data/abaqus_outputs/output.p', 'rb') as fid:
    xyz = pickle.load(fid)
output2 = xyz
'''
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(X, Y, Z, c='b')
x, y, z = output.reshape(nx*ny, 3).T
ax.scatter(x, y, z, c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#print(len(U1))
U = np.sqrt(np.square(U1) + np.square(U2) + np.square(U3))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.reshape(x, (20,50))
Y = np.reshape(y, (20,50))
Z = np.reshape(z, (20,50))
#ax.plot_surface(X, Y, Z)
# xx, yy, zz = np.meshgrid(x,y,z)
# use xyz points from outputs file (before processing) for points
#print(xo.shape, yo.shape, zo.shape, U.shape)
grid_u = griddata(np.array([xo,yo,zo]).T, U, np.array([x,y,z]).T, fill_value=0, rescale=True, method='nearest')
print('hi')
#print(grid_u)
grid = np.array(grid_u)
grid = grid/grid.max()
# print(grid)
G = np.reshape(grid, (20,50))
surf = ax.plot_surface(X, Y, Z, facecolors=cm.jet(G))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(cm.ScalarMappable(cmap=cm.jet))
plt.show()
