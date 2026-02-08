from model import Model
from darts.engines import redirect_darts_output
import matplotlib.pyplot as plt

redirect_darts_output('binary.log')

m = Model()

m.set_reservoir(nx=40, nz=100)

salinity = 0.  # if 0., ions will not be added
temperature = 300.
m.set_physics(components=["H2O", "CO2", "C1"], phases=["Aq", "V"], salinity=salinity, temperature=temperature,
              impurity=0.05, n_points=10001, zero=1e-12)

m.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=36.5, tol_newton=1e-3, tol_linear=1e-6, it_linear=50, it_newton=16)
# m.params.nonlinear_norm_type = m.params.L1
# m.params.linear_type = m.params.cpu_superlu

mix_name = "-".join(comp for comp in m.components)
m.init(itor_type='linear')
m.set_output(output_folder="results_" + mix_name + ("_ions" if salinity else "") + ("_thermal" if m.physics.thermal else ""))
output_props = ['pressure', 'satV', 'xCO2'] if not m.physics.thermal else ['pressure', 'temperature', 'satV', 'xCO2']
output_props += ['x' + comp for comp in m.components[2:]]
lims = {'pressure': [97.5, 102.5], 'temperature': [299., 301.], 'satV': [0., 1.], 'xCO2': [0., 0.025]}
lims.update({'x' + comp: [0., None] for comp in m.components[2:]})

m.output.output_to_plt(output_properties=output_props, timestep=-1, lims=lims)
plt.show()

for j in range(1000):
    # m.data_ts.dt_max = min(float(j+1), 100.)
    m.run(365)

    if (j+1) % 10 == 0:
        m.output.output_to_plt(output_properties=output_props, timestep=-1, lims=lims)
        plt.show()

m.print_timers()
m.print_stat()

# a = 100  # 50
#     mass = (np.sum(rho_aq[m.single_phase_region:] * x[m.single_phase_region:] * 44.01 / 18.015 *
#                    m.reservoir.volume[m.single_phase_region:])) / a * m.reservoir.poro[m.single_phase_region:]
#
#     # calculate the dissolution mass
#     with open('../results/model0.2.in', 'a', encoding='UTF-8') as a1:
#         np.savetxt(a1, np.column_stack([j, mass]), delimiter='    ', fmt='%0.8f')

# output_directory = m.output_folder
# binary_filename = m.output_folder + '/solution.h5'
# m.output_to_vtk(ith_step=1, output_directory=output_directory, binary_filename=binary_filename)
