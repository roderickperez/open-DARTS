import numpy as np
import pandas as pd
from darts.engines import value_vector, redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


filename = 'out'

# define the model
m = Model()

m.set_reservoir(nx=1000)
#m.set_reservoir_unstr()

zero = 1e-12
m.set_physics_geo(n_points=1000)

T_init = 520.
state_pt_init = value_vector([30., 520.])
enth_init = m.physics.property_containers[0].compute_total_enthalpy(state_pt_init)
m.initial_values = {m.physics.vars[0]: state_pt_init[0],
                    m.physics.vars[1]: enth_init
                    }

m.set_sim_params(first_ts=1e-12, mult_ts=2, max_ts=.1, tol_newton=1e-6, tol_linear=1e-8, it_newton=16, it_linear=50)
m.params.nonlinear_norm_type = m.params.L1

#m.params.nonlinear_norm_type = m.params.L1

m.inj_comp = []
m.inj_temp = 320.
m.p_inj = 40.
m.p_prod = 30.

# init the model
m.init()
m.set_output()

volume = np.array(m.reservoir.mesh.volume, copy=False)
poro = np.array(m.reservoir.mesh.poro, copy=False)
print("Pore volume = " + str(sum(volume * poro)))


x = np.cumsum(m.dx)
y = np.linspace(m.reservoir.nz*2+1, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

properties = m.physics.vars + m.output.properties
time, output = m.output.output_properties(output_properties = properties)
nv = m.physics.n_vars
print_props = [properties[0], properties[2], properties[3]] #, properties.index("satA")]

#%%

fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
# for i, ith_prop in enumerate(print_props):
#     #prop = axs[i].pcolormesh(X, Y, output[ith_prop, :].reshape(m.reservoir.nz, m.reservoir.nx))
#     #plt.colorbar(prop, ax=axs[i])
#     axs[i].plot(x, output[ith_prop, :])
#     axs[i].set_title(properties[ith_prop])

plt.savefig('step0.png', format='png')
colors = ['b', 'r', 'g']
fig, axs = plt.subplots(1, len(print_props), figsize=(20, 6)) #, dpi=100, facecolor='w', edgecolor='k')
ii = 0
for t in range(3):
    m.run(365)
    time_data = pd.DataFrame.from_dict(m.output.store_well_time_data())
    m.print_timers()
    m.print_stat()

    timesteps, output = m.output_properties(output_properties=properties, timestep=t+1)

    if t == 0 or t == 2:
        for i, ith_prop in enumerate(print_props):
            # prop = axs[i].pcolormesh(X, Y, output[ith_prop,:].reshape(m.reservoir.nz, m.reservoir.nx))
            # plt.colorbar(prop, ax=axs[i])
            axs[i].plot(x, output[ith_prop][0], label=f'time = {round((t+1), 2)} year', linestyle='-',
                        color=colors[ii])
            #axs[i].set_title(properties[ith_prop])

            axs[i].set_xlabel(r'Distance, m', fontsize=20)
            axs[i].grid(True)
        ii += 1

axs[2].legend(loc='lower right', prop={'size': 14}, framealpha=0.5)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.tight_layout()
axs[0].set_ylabel("pressure, bar")
axs[1].set_ylabel("temperature, K")
axs[1].set_ylabel("steam saturation")

plt.savefig('all_steps.png', format='png')

if 0:
    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    td.to_pickle("darts_time_data.pkl")
    writer = pd.ExcelWriter('time_data.xlsx')
    td.to_excel(writer, sheet_name='Sheet1')
    writer.close()

#plt.show()
