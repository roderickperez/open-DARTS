import numpy as np
import pandas as pd
from darts.engines import value_vector, redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')

def plot_book(m, x, data, save_data=False):
    prefix = m.output_directory
    colors = ['b', 'r', 'g']

    darts_mult = 1.0
    an_mult = 1.0
    add = 0.0
    if data['name'] == 'p':
        y_label = r'Pressure, bar'
        darts_mult = 1#.e+5
        add = 0.0#2.5e-5
        l_loc = 'upper right'
    elif data['name'] == 't':
        y_label = r'Molar fraction of CH$_4$, -'
        l_loc = 'upper right'
    elif data['name'] == 'uy':
        y_label = r'Temperature, K'
        an_mult = -1000.0
        darts_mult = 1000.0
        l_loc = 'upper left'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    time_id = [32, 57]

    for i, t_id in enumerate(time_id):
        if i == 0:
            darts_label = 'DARTS: t = ' + str(round(10.0, 2)) + ' days'
        elif i == 1:
            darts_label = 'DARTS: t = ' + str(round(100.0, 2)) + ' days'
        else:
            darts_label = 'DARTS: t = ' + str(round(data['time'][t_id], 2)) + ' days'
        ax.plot(x, output[ith_prop, :], linestyle='-', color=colors[i], label=darts_label)
        # analytics_label = 'Analytics: y = ' + str(y_cur) + ' m'
        # ax.semilogx(data['analytics'][y_cur][:, 0], an_mult * data['analytics'][y_cur][:, 1], linestyle='-', color=colors[i], label=analytics_label)

    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(r'Height, m', fontsize=20)
    ax.grid(True)
    ax.legend(loc=l_loc, prop={'size': 14 }, framealpha=0.5)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.savefig(prefix + '/' + data['name'] + '_bai.png')
    # plt.show()



filename = 'out'

# define the model
m = Model()

colors = ['b', 'r', 'g']

m.set_reservoir(1000)
#m.set_reservoir_unstr()

zero = 1e-12
m.set_physics(zero, n_points=1001, components=["C1", "H2O"], temperature=None)

m.initial_values = {"pressure": 180.,
                    "C1": 1e-3,
                    "temperature": 350.
                    }
m.inj_comp = [zero]
m.inj_temp = 300.
m.p_inj = 220.
m.p_prod = 180.

m.set_sim_params(first_ts=1e-2, mult_ts=2, max_ts=1, tol_newton=1e-6, tol_linear=1e-8, it_newton=16, it_linear=50)
m.params.nonlinear_norm_type = m.params.L1

# init the model
m.init()
m.set_output(verbose=True)

volume = np.array(m.reservoir.mesh.volume, copy=False)
poro = np.array(m.reservoir.mesh.poro, copy=False)
print("Pore volume = " + str(sum(volume * poro)))

x = np.cumsum(m.dx)
y = np.linspace(m.reservoir.nz*2+1, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

properties = m.physics.vars + m.output.properties
time, output = m.output.output_properties(output_properties = properties)
nv = m.physics.n_vars
print_props = [properties[0] , properties[1], properties[2]] #, properties.index("satA")]

#%%

fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
# for i, ith_prop in enumerate(print_props):
#     #prop = axs[i].pcolormesh(X, Y, output[ith_prop, :].reshape(m.reservoir.nz, m.reservoir.nx))
#     #plt.colorbar(prop, ax=axs[i])
#     axs[i].plot(x, output[ith_prop, :])
#     axs[i].set_title(properties[ith_prop])

#plt.savefig('step0.png', format='png')
fig, axs = plt.subplots(1, len(print_props), figsize=(24, 6)) #, dpi=100, facecolor='w', edgecolor='k')
ii = 0
str = ["pressure, bar", "methane fraction", "temperature, K"]
for t in range(5):
    m.run(365, save_reservoir_data=True, save_well_data=True)
    # time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data = pd.DataFrame.from_dict(m.output.store_well_time_data())
    m.print_timers()
    m.print_stat()

    timesteps, output = m.output.output_properties(timestep=t+1, output_properties=properties)

    if t == 0 or t == 4:
        for i, ith_prop in enumerate(print_props):
            # prop = axs[i].pcolormesh(X, Y, output[ith_prop,:].reshape(m.reservoir.nz, m.reservoir.nx))
            # plt.colorbar(prop, ax=axs[i])
            axs[i].tick_params(axis="x", labelsize=14)
            axs[i].tick_params(axis="y", labelsize=14)
            axs[i].set_ylabel(str[i], fontsize=20)
            axs[i].plot(x, output[ith_prop][0], label=f'time = {round((t+1), 2)} year', linestyle='-', color=colors[ii])
            #axs[i].set_title(properties[ith_prop])


            axs[i].set_xlabel(r'Distance, m', fontsize=20)
            axs[i].grid(True)
        ii += 1

axs[2].legend(loc='lower right', prop={'size': 14}, framealpha=0.5)


fig.tight_layout()
# axs[0].set_ylabel("pressure, bar", fontsize=20)
# axs[1].set_ylabel("methane fraction", fontsize=20)
# axs[2].set_ylabel("temperature, K", fontsize=20)

plt.savefig('all_steps.png', format='png')


td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

#plt.show()


