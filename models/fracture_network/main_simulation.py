# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
import os
import numpy as np
import pandas as pd
import model
from darts.engines import redirect_darts_output
import shutil
from datetime import datetime
from darts.tools.plot_darts import plot_temp_darts
import pickle
from set_case import set_input_data
def run_simulation(case: str):
    input_data = set_input_data(case)
    print('Running simulation for case', input_data['case_name'])

    # resolve the issue "Length of fracture data not equal to number of fracture cells"
    # calc_frac_aper_by_stress returns discretized fractures data
    #from darts.tools.fracture_network.frac_apertures import calc_frac_aper_by_stress
    #input_data['frac_aper'] = calc_frac_aper_by_stress(input_data)

    # Properties for writing to vtk format:
    # output_directory = 'trial_dir'  # Specify output directory here
    output_directory = 'sol_' + input_data['case_name']

    # delete output dir
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        os.mkdir(output_directory)

    redirect_darts_output(os.path.join(output_directory, 'out.log'))

    # Typical structure of the main.py file (which is the actual file that is being run in PyCharm) is the following:
    # 1) Create model object by calling the Model() constructor from the file model.py
    #   --> This model class contains everything related to the model which is run in DARTS
    #   --> From permeability, to porosity, to the physics used in the simulator, as well as the simulation parameters
    m = model.Model(input_data)
    #m.params.linear_type = sim_params.cpu_superlu
    # After constructing the model, the simulator needs to be initialized. The init() class method is called, which is
    # inherited (https://www.python-course.eu/python3_inheritance.php) from the parent class DartsModel (found in
    # darts/models/darts_model.py (NOTE: This is not the same as the__init__(self, **) method which each class (should)
    # have).

    m.init(verbose=True)

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    m.params.max_ts = 20  # Adjust the maximum time-step as desired (this is overwriting the max_ts specified in model.py)
    size_report_step = 30  # Size of the reporting step (when output is writen to .vtk format)
    num_report_steps = 12*50   # Number of reporting steps (see above)
    start_time = 0  # Starting time of the simulation
    end_time = size_report_step * num_report_steps  # End time of the simulation

    # Before starting the simulation, store initial condition also in .vtk format:
    ith_step = 0  # Store initial conditions as ../solution0.vtk

    #num_wells_tot = len(m.reservoir.well_perf_loc[0]) + len(m.reservoir.well_perf_loc[1])  # Specify here how much wells are being used
    # Specify here the number of properties you want to extract (properties are based on selected physics, see model):
    tot_properties_initial_step = 4
    tot_properties = 3

    # Calculate the size of the properties vector:
    #tot_unknws = m.reservoir.discretizer.fracture_cell_count + m.reservoir.discretizer.matrix_cell_count + num_wells_tot*2

    # Allocate and store the properties in an array:
    property_array = np.empty((m.get_pressure(0).size, tot_properties_initial_step))
    property_array[:, 0] = m.get_pressure(0)
    property_array[:, 1] = m.get_saturation(0)
    property_array[:, 2] = m.get_temperature(0)
    dummmy_perm_for_frac_cells = np.zeros(m.reservoir.discretizer.frac_cells_tot)
    property_array[:, 3] = np.hstack((dummmy_perm_for_frac_cells, m.reservoir.discretizer.perm_x_cell))

    # Write to vtk
    m.output_to_vtk(ith_step=0, output_directory=output_directory)

    property_array = np.empty((m.get_pressure(0).size,tot_properties))

    sim_year = 0.
    m.print_range(sim_year)

    # Run over all reporting time-steps:
    for ith_step in range(num_report_steps):
        # Run engine for reporting_step [days]:
        # print('\n---------------------------SELF-PRINT---------------------------')
        # print('Current simulation time: {:f}'.format((ith_step+1)*size_report_step))
        # print('---------------------------SELF-PRINT---------------------------\n')
        m.run(size_report_step)

        # Allocate and store the properties in an array:
        property_array[:,0] = m.get_pressure(0)
        property_array[:,1] = m.get_saturation(0)
        property_array[:,2] = m.get_temperature(0)

        # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
        if ith_step % 20 == 0:
            m.output_to_vtk(ith_step=ith_step+1, output_directory=output_directory)

        sim_year += size_report_step / 365.
        m.print_range(sim_year)
        m.print_range(sim_year, full=1)

    # After the simulation, print some of the simulation timers and statistics,
    # newton iters, etc., how much time spent where:
    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data['Time (years)'] = time_data['time']/365.

    xls_fname = 'time_data.xlsx'
    if os.path.exists(xls_fname):
        ren_fname = os.path.basename(xls_fname) + '_prev.xlsx'
        if os.path.exists(ren_fname):
            os.remove(ren_fname)
        os.renames(xls_fname, ren_fname)
    with pd.ExcelWriter(xls_fname) as writer:
        time_data.to_excel(writer, sheet_name='Sheet1')

    w = m.reservoir.wells[1]
    ax2 = plot_temp_darts(w.name, time_data)
    #plt.show()

    pkl_fname = 'time_data.pkl'
    if os.path.exists(pkl_fname):
        ren_fname = os.path.basename(pkl_fname) + '_prev.pkl'
        if os.path.exists(ren_fname):
            os.remove(ren_fname)
        os.renames(pkl_fname, os.path.basename(pkl_fname) + '_prev.pkl')
    pickle.dump(time_data, open('time_data.pkl', 'wb'))

if __name__ == "__main__":

    t1 = datetime.now()
    print(t1)

    run_simulation('case_1')

    t2 = datetime.now()
    print((t2 - t1).total_seconds())
