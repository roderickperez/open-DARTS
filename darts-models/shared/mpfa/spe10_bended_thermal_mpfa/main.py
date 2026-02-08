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

def run():
    # 'tpfa' - Python discretizer + tpfa super engine
    # 'mpfa' - C++ (new) discretizer + mpfa super engine
    # permeabilitties and heat conductivities are different between 'tpfa' and 'mpfa'

    discr_type = 'mpfa'
    # discr_type = 'tpfa'
    model_folder = 'meshes/data_20_40_40'

    # run(discr_type='mpfa', model_folder='meshes/data_40_80_80')
    redirect_darts_output('darts_log' + '.txt')

    m = Model(discr_type=discr_type, model_folder=model_folder)

    # After constructing the model, the simulator needs to be initialized. The init() class method is called, which is
    # inherited (https://www.python-course.eu/python3_inheritance.php) from the parent class DartsModel (found in
    # darts/models/darts_model.py (NOTE: This is not the same as the__init__(self, **) method which each class (should)
    # have).
    m.init()

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    eps = 1e-6
    size_report_step = 10.0  # Half Size of the reporting step (when output is writen to .vtk format)
    # num_report_steps = int(5.0 / size_report_step)
    max_dt = 2.0
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    first_ts = 1.e-6
    m.params.first_ts = first_ts

    # Properties for writing to vtk format:
    # output_directory = 'trial_dir'  # Specify output directory here
    output_directory = 'sol_cpp_' + discr_type + '_' + model_folder.split('/')[1] + '_{:s}'.format(m.physics_type)
    m.output_directory = output_directory
    # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
    if discr_type == 'mpfa':
        m.reservoir.write_to_vtk(output_directory, m.cell_property + ['perm'], 0, m.physics.engine)
    else:
        m.reservoir.write_to_vtk_old_discretizer(output_directory, m.cell_property, 0, m.physics.engine)

    # Run over all reporting time-steps:
    ith_step = 0
    #for ith_step in range(num_report_steps):
    while m.physics.engine.t < 2000:

        m.run(size_report_step)

        if discr_type == 'mpfa':
            m.reservoir.write_to_vtk(output_directory, m.cell_property, ith_step + 1, m.physics.engine)
        else:
            m.reservoir.write_to_vtk_old_discretizer(output_directory, m.cell_property, ith_step + 1, m.physics.engine)

        ith_step += 1

    m.print_timers()
    m.print_stat()
    
if __name__ == '__main__':
    pass
    #run()
    
