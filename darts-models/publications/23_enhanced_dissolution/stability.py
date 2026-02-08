import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

poro_list = np.array(['model1', 'model2', 'model3', 'model4'])
dx = np.array(['dx=0.2m', 'dx=0.5m', 'dx=1.0m', 'dx=2.0m'])

plt.figure(figsize=(10, 8))
for i in range(len(poro_list)):
    str = '%s.in'%poro_list[i]
    if os.path.isfile(str):
        time = pd.DataFrame(np.genfromtxt(str))[0][:]
        mass = pd.DataFrame(np.genfromtxt(str))[1][:]
        rate = np.zeros(len(time))

        # make the rate interpolation
        former = np.append(0, mass[1:])
        latter = np.append(0, mass[0:-1])
        rate = former - latter
        rate[0] = mass[0]

        plt.rc('font', size=16)
        plt.grid(True, which='both', linestyle='-.')
        plt.tick_params(direction='in', length=1, width=1, colors='k',
                       grid_color='k', grid_alpha=0.2)

        plt.loglog((time+1)*180/360, rate, label='%s'%dx[i])
        # plt.xlim(0.5, 200)
        # plt.ylim(0.5, 5)
        plt.xlabel('t, years')
        plt.ylabel('F, kg/(m$^2$year)')
        # plt.legend(loc='lower left', fontsize=14)
        plt.legend(fontsize=14)
# plt.savefig('convergence.pdf', dpi=100)
plt.show()
