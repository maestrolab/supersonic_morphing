import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib import cm


# Loudness data from loudness_abaqus.py
f = open('../data/loudness/loudness_small_simple_noTE_50S_EqvAtest.p', 'rb')
loudness = pickle.load(f)
f.close()

# "_pickle.UnpicklingError: the STRING opcode argument must be quoted" error,
# convert outputs pickle file to unix file endings using dos2unix.py in data
# folder

# Displacement data from the mid-line
f = open('../data/abaqus_outputs/line_outputs_small_simple_noTE_50S.p', 'rb')  #
line_data = pickle.load(f, encoding='latin1')
f.close()

displacements = {}
temperatures = {}
x_coords = {}
fig = plt.figure()
ax = plt.axes()
# Get the displacement and temperature data along the line for each increment
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(line_data['U'][steps[0]][0][0])
#print(line_data['COORD'][steps[0]][0][0])
#STAAAAHP

temperature = []
for step in steps:
    for inc in range(len(line_data['U'][step])):
        y_sorted = []
        displacements[step] = []
        x_coords[step] = []
        for j in range(len(line_data['U'][step][inc])):
            x_coords[step].append(line_data['COORD'][step][inc][j][0])
            displacements[step].append(np.linalg.norm(line_data['U'][step][inc][j]))
        #print(displacements[step])
        temperature.append(line_data['NT11'][step][inc][0])

        theta = np.linspace(0, 90, len(displacements[step]))

        # Plot the displacement along the line against the angle
        #print(displacements[step])
        theta = np.array(theta)
        y_sorted = [y for x, y in sorted(zip(x_coords[step], displacements[step]))]
        y_sorted = np.array(y_sorted)
        plt.plot(theta, y_sorted, label=str(inc))


#plt.legend()
temperature = np.array(temperature)
t_colors = (temperature-temperature.min())/(temperature.max()-temperature.min())
colormap = plt.cm.Spectral
colors = [colormap(i) for i in t_colors]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])
    j.set_linewidth(1)
colors2 = cm.ScalarMappable(norm=plt.Normalize(temperature.min(),temperature.max()),cmap=cm.Spectral)
colors2.set_array([])
fig.colorbar(colors2)

plt.xlabel('Angle (degrees)')
plt.ylabel('Displacement (m)')
ax.set_xlim(0, 90)
ax.set_ylim(0, 0.0003)
plt.show()
