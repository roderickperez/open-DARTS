# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
from model import Model
from model_geothermal import Model_geothermal
import numpy as np
import meshio
from darts.engines import redirect_darts_output
import time
import pandas as pd
from matplotlib import pyplot as plt

# Run a realization to generate "true" data (otherwise it will be read from pickle file, if it exists)
generate_true_data = True
plot_and_check = generate_true_data  # plot and check the "true" model (observation data); plot and update kernel size, EM data, etc.
plot_and_check = True
true_realization = ""

T = 3000
training_time = 2000
report_step = 50

# 'tpfa' - Python discretizer + tpfa super engine
# 'mpfa' - C++ (new) discretizer + mpfa super engine
# permeabilitties and heat conductivities are different between 'tpfa' and 'mpfa'

# discr_type = 'mpfa'
discr_type = 'tpfa'

# model_folder = 'meshes/data_20_40_40'
model_folder = 'meshes/data_40_80_80'

physics_type = "geothermal"

training_model = True  # switch off to compare and plot the optimized results and un-optimized result
optimization = False  # switch off to compare the adjoint and numerical gradient
apply_adjoint_method = True  # switch off to apply numerical method

add_prod_rate_to_objfun = True
add_inj_rate_to_objfun = False
add_BHP_to_objfun = False
add_well_tempr_to_objfun = False
add_temperature_to_objfun = False
add_customized_op_to_objfun = False

# if switch on, the customized operator should be defined in "set_op_list", e.g. temperature
customize_new_operator = add_temperature_to_objfun


if generate_true_data:
    starting_time = time.time()

    redirect_darts_output('darts_log' + '_' + discr_type + '_' + model_folder.split('/')[1] + '_' + physics_type + '.txt')

    # m = Model(discr_type=discr_type, model_folder=model_folder, T=T, report_step=report_step,
    #           customize_new_operator=customize_new_operator)

    true_model = Model_geothermal(discr_type=discr_type, model_folder=model_folder, T=T, report_step=report_step,
                                  physics_type=physics_type, customize_new_operator=customize_new_operator)

    # After constructing the model, the simulator needs to be initialized. The init() class method is called, which is
    # inherited (https://www.python-course.eu/python3_inheritance.php) from the parent class DartsModel (found in
    # darts/models/darts_model.py (NOTE: This is not the same as the__init__(self, **) method which each class (should)
    # have).
    true_model.init()

    run(true_model, export_to_vtk=True)
    true_model.print_timers()
    true_model.print_stat()

    ending_time = time.time()
    print('Total elapsed %s sec' % (ending_time - starting_time))
    
    # Save realization output as a true data
    time_data = pd.DataFrame.from_dict(true_model.physics.engine.time_data)
    time_data.to_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
    time_data_report = pd.DataFrame.from_dict(true_model.physics.engine.time_data_report)
    time_data_report.to_pickle("_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))

    if customize_new_operator:
        # time-lapse temperature data
        time_data_customized = np.array(true_model.physics.engine.time_data_customized)[:,
                               0:true_model.reservoir.mesh.n_res_blocks]
        time_data_report_customized = np.array(true_model.physics.engine.time_data_report_customized)[:,
                                      0:true_model.reservoir.mesh.n_res_blocks]
        np.savetxt('_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step),
                   np.array(time_data_customized))
        np.savetxt('_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step),
                   np.array(time_data_report_customized))
else:
    time_data = pd.read_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
    time_data_report = pd.read_pickle(
        "_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))
    if customize_new_operator:
        time_data_customized = np.loadtxt(
            '_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step))
        time_data_report_customized = np.loadtxt(
            '_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step))





# plot and check the TRUE results
if plot_and_check:
    # plot well curves--------------------------------------------------------------------------------------------------
    df_data = time_data.copy()
    df_data_report = time_data_report.copy()

    doublet_idx = 1
    time_arr = time_data['time'].to_numpy()
    time_report_arr = time_data_report['time'].to_numpy()

    plt.subplot(2, 4, 1)
    plt.plot(time_arr, time_data['INJ%s : temperature (K)' % doublet_idx].to_numpy() - 273.15)
    plt.title('temperature (C)')
    plt.ylabel('I%s' % doublet_idx)
    plt.subplot(2, 4, 2)
    plt.plot(time_arr, time_data['INJ%s : BHP (bar)' % doublet_idx].to_numpy())
    plt.title('BHP (bar)')
    plt.subplot(2, 4, 3)
    plt.plot(time_arr, time_data['INJ%s : water rate (m3/day)' % doublet_idx].to_numpy())
    plt.title('water rate (m3/day)')
    plt.subplot(2, 4, 4)
    plt.plot(time_report_arr, time_data_report['INJ%s : water  acc volume (m3)' % doublet_idx].to_numpy())
    plt.title('water acc volume (m3)')

    plt.subplot(2, 4, 5)
    plt.plot(time_arr, time_data['PRD%s : temperature (K)' % doublet_idx].to_numpy() - 273.15)
    plt.ylabel('P%s' % doublet_idx)
    plt.subplot(2, 4, 6)
    plt.plot(time_arr, time_data['PRD%s : BHP (bar)' % doublet_idx].to_numpy())
    plt.subplot(2, 4, 7)
    plt.plot(time_arr, time_data['PRD%s : water rate (m3/day)' % doublet_idx].to_numpy())
    plt.subplot(2, 4, 8)
    plt.plot(time_report_arr, time_data_report['PRD%s : water  acc volume (m3)' % doublet_idx].to_numpy())

    plt.show()


def run(self, export_to_vtk):
    import random
    # use seed to generate the same random values every run
    random.seed(3)
    if export_to_vtk:
        # Properties for writing to vtk format:
        # output_directory = 'trial_dir'  # Specify output directory here
        output_directory = 'sol_cpp_' + self.discr_type + '_' + self.model_folder.split('/')[1] + '_{:s}'.format(
            self.physics_type)
        self.output_directory = output_directory
        # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
        if self.discr_type == 'mpfa':
            self.reservoir.write_to_vtk(output_directory, self.cell_property + ['perm'] + ["depth"], 0, self.engine)
        else:
            self.reservoir.write_to_vtk_old_discretizer(output_directory, self.cell_property + ["depth"], 0,
                                                        self.engine)

    ith_step = 0
    # now we start to run for the time report--------------------------------------------------------------
    time_step = self.report_step
    even_end = int(self.T / time_step) * time_step
    time_step_arr = np.ones(int(self.T / time_step)) * time_step
    if self.T - even_end > 0:
        time_step_arr = np.append(time_step_arr, self.T - even_end)

    for ts in time_step_arr:
        for i, w in enumerate(self.reservoir.wells):
            if w.name[:3] == 'PRD':
                w.control = self.physics.new_bhp_prod(self.p_init - 10)
                # w.control = self.physics.new_rate_prod(self.water_rate, 0)
            elif w.name[:3] == 'INJ':
                # w.control = self.physics.new_rate_inj(200, self.inj, 1)
                w.control = self.physics.new_bhp_inj(self.p_init + 10, self.inj)
                # w.control = self.physics.new_rate_inj(self.water_rate, self.inj, 0)
                # w.control = self.physics.new_bhp_inj(450, self.inj)

        self.run_python(ts)
        # self.engine.run(ts)
        self.physics.engine.report()
        if export_to_vtk:
            if self.discr_type == 'mpfa':
                self.reservoir.write_to_vtk(output_directory, self.cell_property, ith_step + 1, self.physics.engine)
            else:
                self.reservoir.write_to_vtk_old_discretizer(output_directory, self.cell_property, ith_step + 1,
                                                            self.physics.engine)

            ith_step += 1
    return
