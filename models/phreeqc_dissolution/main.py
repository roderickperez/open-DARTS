import os
os.environ["OMP_NUM_THREADS"] = "1"
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

def run(domain, max_ts, nx=100):
    redirect_darts_output('log.txt')

    # Create model
    m = Model(domain=domain, nx=nx)
    m.sol_filename = 'nx' + str(nx) + '.h5'

    # Initialize simulations
    m.init()

    # Initialization check
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, 39]
    volume = np.array(m.reservoir.mesh.volume, copy=False)
    total_pv = np.sum(volume[:m.reservoir.n] * poro) * 1e6
    print('Total pore volume:', total_pv, 'cm3')
    print('Injection rate:', m.inj_rate / total_pv * 1e+6, 'PV/day')

    m.params.max_ts = max_ts

    if domain == '1D':
        plot_profiles(m)
        m.run(days=0.001, restart_dt=max_ts)
        plot_profiles(m)
        m.run(days=0.004, restart_dt=max_ts)
        plot_profiles(m)
        m.run(days=0.005, restart_dt=max_ts)
        plot_profiles(m)
    else:
        m.output_to_vtk(ith_step=0)

    for i in range(24):
        m.run(days=1. / 24, restart_dt=max_ts)
        if i > 0: m.params.first_ts = max_ts
        if domain == '1D':
            plot_profiles(m)
        else:
            m.output_to_vtk(ith_step=i + 1)

    # Print some statistics
    print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')

    m.print_timers()
    m.print_stat()

def plot_profiles(m):
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, 39]

    n_plots = 3
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(6, 11))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]

    ax[0].plot(x, Xm[:, 0], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = ['b', 'r', 'g', 'm', 'cyan']

    for i in [1,4]: # Solid / O
        label = r'$\mathrm{CaCO_3}$(s)' if i == 1 else m.physics.vars[i]
        ax[1].plot(x, Xm[:, i], color=colors[i-1], label=label)
    for i in [2,3]: # Ca / C
        ax1.plot(x, Xm[:, i], color=colors[i-1], label=m.physics.vars[i])
    ax[1].plot(x, 1.0 - np.sum(Xm[:, 1:], axis=1), color=colors[n_vars - 1], label=m.physics.property_containers[0].components_name[-1])
    ax[2].plot(x, poro, color=colors[0], label='porosity')

    t = round(m.physics.engine.t * 24, 4)
    ax[0].text(0.21, 0.9, 'time = ' + str(t) + ' hours',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[n_plots - 1].set_xlabel('distance, m', fontsize=16)
    ax[1].set_ylabel(r'\textcolor{blue}{z$_{CaCO3}$(s)}, \textcolor{magenta}{z$_O$} and \textcolor{cyan}{z$_H$}', fontsize=20)
    ax[2].set_ylabel('porosity', fontsize=16)
    ax1.set_ylabel(r'\textcolor{red}{z$_{Ca}$} and \textcolor{green}{z$_C$}', fontsize=20)

    ax[0].set_ylim(99.99, 100.3)
    ax[1].set_ylim(-0.01, 1.01)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.01)
    ls = 14
    ax[0].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'1d_dissolution_time_{t}.png', dpi=300)
    # plt.show()
    plt.close(fig)

def write_2d_output_for_paper(paths):
    import pyvista as pv

    m = Model(domain='2D', nx=100)
    m.init()
    nv = m.physics.n_vars
    ncells = m.reservoir.n

    for path in paths:
        # Read the VTK file using pyvista
        mesh = pv.read(path)

        # Check if there is cell data and modify it
        if mesh.cell_data:
            # add hydrogen
            mesh.cell_data['H'] = 1.0 - mesh.cell_data['O'] - mesh.cell_data['Ca'] - mesh.cell_data['C']

            # add porosity
            X = np.asarray(m.physics.engine.X)
            for k, var in enumerate(m.physics.vars):
                X[k:nv*ncells:nv] = mesh.cell_data[var]
            op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
            poro = op_vals[:m.reservoir.mesh.n_res_blocks, 39]
            mesh.cell_data['porosity'] = poro

        # Create a new file name for the output
        output_path = path.replace(".vts", "_modified.vts")

        # Write the modified mesh back to a new VTK file
        mesh.save(output_path)

# paths = ['./100x100/data_ts3.vts',
#          './100x100/data_ts14.vts']
# write_2d_output_for_paper(paths=paths)

# 1D
# run(domain='1D', nx=100, max_ts=1.e-2)
run(domain='1D', nx=200, max_ts=1.e-3) # 4.e-5)
# run(domain='1D', nx=500, max_ts=5.e-4)

# 2D
# run(domain='2D', nx=10, max_ts=1.e-2)