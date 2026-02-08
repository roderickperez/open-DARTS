# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
from model import Model
import numpy as np
import meshio
from darts.engines import redirect_darts_output

try:
    # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
    from darts.engines import set_num_threads
    set_num_threads(1)
except:
    pass

def run(discr_type, mesh_file):
    redirect_darts_output('darts_log' + '.txt')

    m = Model(discr_type=discr_type, mesh_file=mesh_file)

    # After constructing the model, the simulator needs to be initialized. The init() class method is called, which is
    # inherited (https://www.python-course.eu/python3_inheritance.php) from the parent class DartsModel (found in
    # darts/models/darts_model.py (NOTE: This is not the same as the__init__(self, **) method which each class (should)
    # have).
    m.init()
    m.set_output()

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    eps = 1e-6
    size_report_step = 5000.0  # Half Size of the reporting step (when output is writen to .vtk format)
    m.data_ts.dt_max = 2 * 365.0
    m.data_ts.first_ts = 1.0

    # Properties for writing to vtk format:
    # output_directory = 'trial_dir'  # Specify output directory here
    output_directory = 'sol_cpp_' + discr_type + '_{:s}'.format(m.physics_type)
    # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
    if discr_type == 'mpfa':
        m.reservoir.write_to_vtk(output_directory, m.physics.vars, 0, m.physics)
    else:
        tot_unknws = m.reservoir.unstr_discr.fracture_cell_count + m.reservoir.unstr_discr.matrix_cell_count + len(m.reservoir.wells) * 2
        tot_properties = 2
        pressure_field = m.physics.engine.X[:-1:2]
        saturation_field = m.physics.engine.X[1::2]
        property_array = np.empty((tot_unknws, tot_properties))
        property_array[:, 0] = pressure_field
        property_array[:, 1] = saturation_field
        m.reservoir.unstr_discr.write_to_vtk(output_directory, property_array, m.physics.vars, 0)

    # Run over all reporting time-steps:
    ith_step = 0
    #for ith_step in range(num_report_steps):
    while m.physics.engine.t < 30000:

        m.run(days=size_report_step)

        if discr_type == 'mpfa':
            m.reservoir.write_to_vtk(output_directory, m.physics.vars, ith_step + 1, m.physics)
        else:
            pressure_field = m.physics.engine.X[:-1:2]
            saturation_field = m.physics.engine.X[1::2]
            property_array = np.empty((tot_unknws, tot_properties))
            property_array[:, 0] = pressure_field
            property_array[:, 1] = saturation_field
            m.reservoir.unstr_discr.write_to_vtk(output_directory, property_array, m.physics.vars, ith_step+1)

        ith_step += 1

    # After the simulation, print some of the simulation timers and statistics,
    # newton iters, etc., how much time spent where:
    m.print_timers()
    m.print_stat()

## TPFA super-engine
# structured
# run(discr_type='tpfa', mesh_file='meshes/column_1d_tpfa.msh')
# unstructured
# run(discr_type='tpfa', mesh_file='meshes/column_tetra_tpfa.msh')
## MPFA super-engine
# structured
# run(discr_type='tpfa', mesh_file='meshes/column_1d.msh')
# unstructured
run(discr_type='mpfa', mesh_file='meshes/ccdfm_hydrocoin.msh')