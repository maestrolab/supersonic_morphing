import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import pickle
from scipy.signal import savgol_filter
from cycler import cycler


mid_output_name = 'mid_outputs_small_simple_noTE_50S'
loudness_filenames = ['loudness_small_simple_fix1_noTE_50S_weather1',
                      'loudness_small_simple_fix1_noTE_50S_weather2',
                      'loudness_small_simple_fix1_noTE_50S_weather3',
                      'loudness_small_simple_fix1_noTE_50S_weather4']

legend_labels = ['34 lat, -118 lon',
                 '36 lat, -105 lon',
                 '38 lat, -93 lon',
                 '40 lat, -80 lon']

color = ['r','b']
colors = ['m', 'g', 'c', 'y']
marker = ['', '--', '.-', '-.']

plt.figure()
for j in range(len(loudness_filenames)):

    f = open('../data/loudness/' + loudness_filenames[j] + '.p', 'rb')
    loudness = pickle.load(f)
    f.close()
    f = open('../data/abaqus_outputs/' + mid_output_name + '.p', 'rb')
    mid_data = pickle.load(f, encoding='latin1')
    f.close()

    displacements = {}
    temperatures = {}
    steps = ['Step-2', 'Step-3']
    U0 = np.linalg.norm(mid_data['U'][steps[0]][0])
    k = 0
    for step in steps:
        displacements[step] = []
        temperatures[step] = []

        for i in range(len(mid_data['U'][step])):
            displacements[step].append(np.linalg.norm(mid_data['U'][step][i]))# - U0)
            temperatures[step].append(mid_data['NT11'][step][i])

        plt.plot(temperatures[step][:len(loudness[step])],
                 loudness[step], colors[j])
        k += 1
    plt.plot([], [], colors[j], label=legend_labels[j])

plt.legend(loc='upper left')
plt.xlabel('Temperature (K)')
plt.ylabel('Loudness (PLdB)')
plt.show()

# Basemap to plot locations
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(12, 6))

lon = [-118, -105, -93, -80]
lat = [34, 36, 38, 40]

m = Basemap(projection='merc', llcrnrlat=13, urcrnrlat=58,
            llcrnrlon=-144, urcrnrlon=-53, resolution='c')
map_lon, map_lat = m(*(lon, lat))

m.drawstates()
m.drawcountries(linewidth=1.0)
m.drawcoastlines()
m.scatter(map_lon, map_lat, marker='D', color='m')
'''
for i, txt in enumerate(legend_labels):
    plt.annotate(txt, (map_lon[i], map_lat[i]), color='m')
'''
plt.show()
