# Equivalent Area distribution
import os
import numpy as np
import matplotlib.pyplot as plt

eqa_filename = 'mach1p600_aoa0p000_phi00p00.eqarea'

# import equivalent area from file
data_dir = os.path.join(os.path.dirname(__file__), "..","..","rapidboom","misc")

equiv_area_dist = np.genfromtxt(os.path.join(data_dir,
                                             eqa_filename))

position = equiv_area_dist[:, 0]
area = equiv_area_dist[:, 1]

# plot new equivalent area
plt.plot(position, area)
#plt.plot(self.new_equiv_area[:,0], (self.new_equiv_area[:,1])/(10.7639))
plt.title('Gaussian change in $A_E$$_q$, Amplitude: 0.03 $m^2$, Standard deviation: 0.5 m, Location: 30.0 m', fontsize = 16)
plt.xlabel("Axial position(m)", fontsize = 16)
plt.ylabel('$A_E$$_q$ ($m^2$)', fontsize = 16)
#plt.legend(['Baseline $A_E$$_q$', 'Modified $A_E$$_q$'], fontsize = 16)
plt.xlim((0, 50))
plt.show()
