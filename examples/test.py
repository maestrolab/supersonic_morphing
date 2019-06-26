import matplotlib.pyplot as plt
import numpy as np
import pickle

f = open('../data/loudness/loudness_small_simple_test1.p', 'rb')
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

displacements = {}
temperatures = {}
steps = ['Step-2', 'Step-3']
mid_index = 5#int(len(data['U'][steps[0]][0])/2)
print(mid_index)
U0 = data['U'][steps[0]][0][mid_index]
print(U0)
print(data['U']['Step-2'][0][mid_index])
print(data['U'].keys())
for i in range(len(data['U'][steps[0]][0])):
    '''
    print(i)
    print(' - ')
    print(data['U'][steps[0]][0][i])
    print('\n')
    '''
    if abs(data['U'][steps[0]][0][i][0] + 6.63482069e-06) < 1e-14:
        print(i)
        print('yay')
'''
for step in steps:
    print(step)
    displacements[step] = []
    temperatures[step] = []

    for i in range(len(data['U'][step])):
        print(i, data['U'][step][i][mid_index], data['U'][step][i][mid_index])
        displacements[step].append(np.linalg.norm(data['U'][step][i][mid_index]) ) # - U0
        temperatures[step].append(data['NT11'][step][i][mid_index])


label2='Cooling'
label3='Heating'
        
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

plt.figure()
plt.plot(temperatures['Step-2'],
         displacements['Step-2'], 'b', label='Cooling')
plt.plot(temperatures['Step-3'],
         displacements['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()


plt.figure()
plt.plot(temperatures['Step-2'],
         loudness['Step-2'], 'b', label='Cooling')
plt.plot(temperatures['Step-3'],
         loudness['Step-3'], 'r', label='Heating')
plt.legend()
plt.show()

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