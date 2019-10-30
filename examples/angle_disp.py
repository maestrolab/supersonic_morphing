import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import pickle


# Loudness data from loudness_abaqus.py
f = open('../data/loudness/loudness_small_simple_fix1_noTE_50S_EqvAtest.p', 'rb')
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
plt.figure()
# Get the displacement and temperature data along the line for each increment
steps = ['Step-2', 'Step-3']
U0 = np.linalg.norm(line_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(line_data['U'][step])):
        displacements[step].append(np.linalg.norm(
                                   line_data['U'][step][i]))  # - U0)
        temperatures[step].append(line_data['NT11'][step][i])

    theta = np.linspace(0, 90, len(displacements[step]))
    # Plot the displacement along the line against the angle
    plt.plot(theta, displacements[step])

plt.show()
