from darts.engines import value_vector, redirect_darts_output
from model import Model
import gzip, shutil
import numpy as np

# from darts.engines import set_num_threads
# set_num_threads(1)

# with open('SPE10_input_ori.txt', 'rb') as f_in:
#     with gzip.open('SPE10.txt.gz', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

with gzip.open('SPE10.txt.gz', 'rb') as f_in:
    with open('SPE10_input.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

redirect_darts_output('SPE10.log')
m = Model()

m.set_sim_params(first_ts=1e-1, mult_ts=2.5, max_ts=1000, tol_newton=1e0, tol_linear=1e-2, it_newton=10)
# eta constrain dt_mult < mult_ts to provide target max dX change evaluated at last timestep 
m.data_ts.eta[:] = 0.6  # here we defined all variables max change in dX = 0.6 
m.data_ts.eta[0] = 100  # ... and allow pressure max change to be 100

m.init()
m.set_output(output_folder='output/SPE10')

NT = 5
for i in range(NT):
    m.run(200, verbose=True)
m.print_timers()
m.print_stat()
m.output.output_to_vtk()

