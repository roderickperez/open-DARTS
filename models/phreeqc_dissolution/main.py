# import os
# os.environ["OMP_NUM_THREADS"] = "1"
from model import Model
from darts.engines import redirect_darts_output
import pandas as pd
import numpy as np

# matplotlib
import matplotlib
matplotlib.use('pgf')
matplotlib.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{color}')
from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=16)

def run(max_ts, nx=100):
    redirect_darts_output('log.txt')

    # Create model
    m = Model(nx=nx)
    m.sol_filename = 'nx' + str(nx) + '.h5'

    # Initialize simulations
    m.init()

    # Initialization check
    poro = m.evaluate_porosity()
    volume = np.array(m.reservoir.mesh.volume, copy=False)
    total_pv = np.sum(volume[:m.reservoir.n] * poro) * 1e6
    print('Total pore volume:', total_pv, 'cm3')

    m.params.max_ts = max_ts
    # plot_profiles(m)
    m.run(0.01)
    # plot_profiles(m)
    for i in range(8):
        m.run(days=0.1, restart_dt=max_ts)
        if i > 0: m.params.first_ts = max_ts
        plot_profiles(m)

    # Print some statistics
    print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')

    m.print_timers()
    m.print_stat()

def plot_profiles(m):
    Xm = np.copy(m.physics.engine.X[:m.reservoir.n*m.physics.nc])
    poro = m.evaluate_porosity()

    n_plots = 3
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(6, 11))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    ax[0].plot(x, Xm[0:n_cells*n_vars:n_vars], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = ['b', 'r', 'g', 'm', 'y', 'orange']

    for i in [1,4]: # Solid / O
        ax[1].plot(x, Xm[i:n_cells*n_vars:n_vars], color=colors[i-1], label=m.physics.vars[i])
    for i in [2,3]: # Ca / C
        ax1.plot(x, Xm[i:n_cells*n_vars:n_vars], color=colors[i-1], label=m.physics.vars[i])
    ax[2].plot(x, poro, color=colors[n_vars], label='porosity')

    t = round(m.physics.engine.t, 4)
    ax[0].text(0.21, 0.9, 'time = ' + str(t) + ' days',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[n_plots - 1].set_xlabel('distance, x', fontsize=16)
    ax[1].set_ylabel(r'\textcolor{blue}{Solid} and \textcolor{magenta}{O} concentrations', fontsize=16)
    ax[2].set_ylabel('porosity', fontsize=16)
    ax1.set_ylabel(r'\textcolor{red}{Ca} and \textcolor{green}{C} concentrations', fontsize=16)
    # ax[1].set_ylabel(r'\textcolor[rgb]{0,0,1}{Solid} and \textcolor[rgb]{1,0,0}{O} concentrations', fontsize=16)

    ax[0].set_ylim(99.99, 100.3)
    ax[1].set_ylim(-0.01, 0.5)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.03)
    ax[0].legend(loc='upper right', prop={'size': 16}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': 16}, framealpha=0.9)
    ax[2].legend(loc='upper right', prop={'size': 16}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': 16}, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'time_{t}.png')

    plt.show()

# run(nx=100, max_ts=1.e-2)
run(nx=200, max_ts=1.e-2)
# run(nx=500, max_ts=4.e-3)