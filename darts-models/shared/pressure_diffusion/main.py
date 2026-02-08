import numpy as np
import pandas as pd
import os

from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.tools.logging import redirect_all_output, abort_redirection
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props
from matplotlib import cm


if __name__ == '__main__':

    redirect_all_output('run.log', append=False)
    n = Model()
    n.set_sim_params(first_ts=0.1, mult_ts=4, max_ts=3650, tol_newton=1e-6, tol_linear=1e-8)
    n.init()
    n.set_output(save_initial=False)
    n.run(1, 0, False, True, False)

    X = np.array(n.physics.engine.X, copy=False)
    nb = n.reservoir.mesh.n_res_blocks
    plt.figure(num=1, figsize=(12, 8), dpi=100)
    plt.subplot(611)
    plt.plot(X[0:nb*2:2])

    w = n.reservoir.wells[0]
    from darts.engines import well_control_iface
    n.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                is_inj=False, target=0.)    
    for i in range(5):
        n.run(10*365, 0, False, False, False)
        plt.subplot(610 + (i + 2))
        plt.plot(X[0:nb*2:2])

    plt.savefig('out.png')
    #n.print_timers()
    n.print_stat()


