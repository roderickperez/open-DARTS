import numpy as np
import pandas as pd

from model import Model
from darts.engines import value_vector, redirect_darts_output

GRAV = '_grav'
grid_1D = True
for res in [1]:
    redirect_darts_output('run' + str(res) + '.log')
    n = Model(grid_1D=grid_1D, res=res)
    n.init()
    n.params.max_ts = 1e-0

    if grid_1D:
        filename_base = 'DARTS_11' + GRAV
    else:
        filename_base = 'DARTS_21' + GRAV

    n.run_python(50)
    n.save_restart_data()
    n.print_timers()
    n.print_stat()

    if grid_1D:
        n.print_and_plot_1D(f'{filename_base}_t1000')
    else:
        n.print_and_plot_2D()
