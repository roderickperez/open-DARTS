import numpy as np
from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
import pandas as pd
import os


redirect_darts_output('binary.log')

m = Model()

""" DEFINE RESERVOIR """
# region Reservoir

dr = np.array(6*[0.05] + 7*[0.1] + 10*[0.2] + 12*[0.5] + 14*[1] + [2, 5, 10, 20, 50, 100])
(nr, nz) = (len(dr), 100)
poro = np.ones((nr, nz)) * 0.35
perm = np.ones((nr, nz)) * 2000

m.set_reservoir(nr=nr, dr=dr, nz=nz, dz=0.2, poro=poro.flatten(order='F'), perm=perm.flatten(order='F'))

""" DEFINE PHYSICS """
salinity = 0.0  # if 0., ions will not be added
zero = 1e-12
n_points = 401

m.set_physics_VAq(components=["CO2", "H2O"], phases=["Aq", "V"], salinity=salinity, temperature=None,
                  vl_phases=False, zero=zero, n_points=n_points)

m.p_init = 84
m.p_inj = m.p_init + 10
m.t_init = 320
m.t_inj = 280
m.inj_stream = [1 - zero]
m.initial_values = { "pressure": m.p_init,
                    "CO2": zero,
                    "temperature": m.t_init,
                    }

timesteps = [0.1, 0.9, 30] + [31] * 11
props = ['satV', 'rho_g', 'CO2_Aq', 'H2O_V']

m.ms_well_flag = False

""" INITIALIZE MODEL """
m.set_sim_params(first_ts=1e-10, mult_ts=4, max_ts=5., tol_newton=1e-4, tol_linear=1e-6, it_newton=8, it_linear=50)

mix_name = "-".join(comp for comp in m.components)
m.init()
output_folder = "results_" + mix_name + ("-{:.0f}".format(m.p_init)) + ("_ions" if salinity else "")
m.set_output(output_folder=output_folder)

""" DEFINE OUTPUT """
output_props = ['pressure', 'temperature'] + props
# output_props += ['y' + comp for comp in m.components[2:]]
lims = {'pressure': [m.p_init, m.p_inj], 'temperature': [m.t_inj - 5., m.t_init + 5.], 'satV': [0, 1.]}
lims.update({'satLCO2': [0., 1.]})
aspect = 'equal'  # 'equal', 'auto' or float
cmap = 'RdBu_r'
logx = True

timestep, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                          aspect_ratio=aspect, logx=logx, cmap=cmap)
m.output_to_vtk(ith_step=0, output_properties=output_props)
plt.savefig(output_folder + '/figures/fig0.png')

# exit(1)
""" RUN MODEL """
data_dt = None
m.run = super(Model, m).run  # uses base DartsModel method
m.run_timestep = super(Model, m).run_timestep  # uses base DartsModel method
m.data_ts.eta[-1] = 3

for j, ts in enumerate(timesteps):
    m.run(ts)

    if (j+1) % 1 == 0:
        timestep, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

        m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                                  aspect_ratio=aspect, logx=logx, cmap=cmap)
        m.output_to_vtk(ith_step=j+1, output_properties=output_props)
        plt.savefig(output_folder + '/figures/fig' + str(j+1) + '.png')


time_data_dict = m.output.store_well_time_data()
time_data_df = pd.DataFrame.from_dict(time_data_dict)
time_data_df.to_pickle(os.path.join(m.output_folder, "well_time_data.pkl"))  # as a pickle file

writer = pd.ExcelWriter(os.path.join(m.output_folder, "well_time_data.xlsx"))  # as an excel file
time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()

m.print_timers()
m.print_stat()
