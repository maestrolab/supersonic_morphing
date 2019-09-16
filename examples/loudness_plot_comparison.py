import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import pickle
from scipy.signal import savgol_filter


f = open('../data/loudness/loudness_small_simple_alt6.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt6.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()

displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

# initializing plot and plotting first dataset (alt1)
plt.figure()
plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], 'b', label='Max Disp = 0.2700 cm')

# Getting and plotting data from fifth alternate
f = open('../data/loudness/loudness_small_simple_alt1.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt1.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()
displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], label='Max Disp = 0.2690 cm')

# Getting and plotting data from second alternate
f = open('../data/loudness/loudness_small_simple_alt2.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt2.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()
displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], label='Max Disp = 0.2635 cm')

# Getting and plotting data from third alternate
f = open('../data/loudness/loudness_small_simple_alt3.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt3.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()
displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], label='Max Disp = 0.2614 cm')

# Getting and plotting data from fourth alternate
f = open('../data/loudness/loudness_small_simple_alt4.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt4.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()
displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], label='Max Disp = 0.2574 cm')

# Getting and plotting data from fifth alternate
f = open('../data/loudness/loudness_small_simple_alt5.p', 'rb')
loudness = pickle.load(f)
f.close()
f = open('../data/abaqus_outputs/mid_outputs_small_simple_alt5.p', 'rb')  #
mid_data = pickle.load(f, encoding='latin1')
f.close()
displacements = {}
temperatures = {}
steps = ['Step-2']#, 'Step-3']
U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
for step in steps:
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(mid_data['U'][step])):
        displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
        temperatures[step].append(mid_data['NT11'][step][i])

plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         loudness['Step-2'], label='Max Disp = 0.2524 cm')


plt.xlabel('Temperature (K)')
plt.ylabel('Loudness (PLdB)')
plt.title('Comparison of Loudness Spikes for Similar Abaqus Models')
plt.legend()
plt.show()

'''
# Smoothing attempts
plt.figure()
yhat = savgol_filter(loudness['Step-2'], 11, 3)
plt.plot(temperatures['Step-2'][:len(loudness['Step-2'])],
         yhat, 'b', label='Cooling')
plt.title('Filtered Data')
plt.show()
'''
