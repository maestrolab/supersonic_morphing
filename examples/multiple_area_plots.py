import matplotlib.pyplot as plt
import numpy as np
import pickle

'''
Short script to make multiple plots of area vs distance for figure using data
generated in loudness_abaqus.py for different step-increments
'''

# Step 2-8
with open('../data/images/3Dpicture_test6_4.p', 'rb') as fid:
    pic_data = pickle.load(fid)
# points from Abaqus
A = pic_data['A']
y0_list = pic_data['y0_list']


plt.figure(figsize=(12,6))
plt.plot(y0_list, A, label='  Step 2-8: 82.50 PLdB')
plt.ylabel('Area along Mach Cone')
plt.xlabel('Distance along aircraft')


# Step 2-9
with open('../data/images/3Dpicture_test6_3.p', 'rb') as fid:
    pic_data = pickle.load(fid)
# points from Abaqus
A = pic_data['A']
y0_list = pic_data['y0_list']

plt.plot(y0_list, A, label='  Step 2-9: 82.62 PLdB')


# Step 2-10
with open('../data/images/3Dpicture_test6_2.p', 'rb') as fid:
    pic_data = pickle.load(fid)
# points from Abaqus
A = pic_data['A']
y0_list = pic_data['y0_list']

plt.plot(y0_list, A, label="Step 2-10: $\\bf{79.76 PLdB}$")


# Step 2-11
with open('../data/images/3Dpicture_test6_5.p', 'rb') as fid:
    pic_data = pickle.load(fid)
# points from Abaqus
A = pic_data['A']
y0_list = pic_data['y0_list']

plt.plot(y0_list, A, label='Step 2-11: 82.74 PLdB')


plt.legend()
plt.show()
