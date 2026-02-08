from model import Model
from darts.engines import redirect_darts_output

GRAV = '_grav'
grid_1D = 1  # change to 0 for running 2D version of the benchmark
for res in [2]:  # can run in different resolutions - 2 is default for the paper (120 x 48)
    redirect_darts_output('run' + str(res) + '.log')
    n = Model(grid_1D=grid_1D, res=res, custom_physics=0)
    n.init()
    n.set_output(verbose = True)
    n.params.max_ts = 1e-0

    n.run(1000, save_reservoir_data=False, save_well_data=False)
    # n.save_restart_data()
    n.output.save_data_to_h5('reservoir')
    n.print_timers()
    n.print_stat()

    if grid_1D:
        n.print_and_plot_1D()
    else:
        n.print_and_plot_2D()
