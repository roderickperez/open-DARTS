import os
os.environ["OMP_NUM_THREADS"] = "1"
from model import Model
from darts.engines import redirect_darts_output
import pandas as pd
import numpy as np
import re
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
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]
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

def plot_profiles(m, plot_kinetics=False):
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]

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

    if plot_kinetics:
        evaluator = m.physics.reservoir_operators[0]
        op_kin_rates = op_vals[:m.reservoir.mesh.n_res_blocks, \
                       evaluator.KIN_OP:evaluator.KIN_OP + n_vars]
        op_sr = op_vals[:m.reservoir.mesh.n_res_blocks, 40]
        op_actHp = op_vals[:m.reservoir.mesh.n_res_blocks, 41]

        ms = 4
        n = 5
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))

        kin_rates, SR, actHp = np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells)
        for i in range(n_cells):
            _, _, _, rho_phases, kin_state, _ = m.physics.reservoir_operators[0].property.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1]
            nu_a = 1 - nu_s
            rho_s = m.physics.reservoir_operators[0].property.density_ev['solid'].evaluate(Xm[i, 0]) / m.physics.reservoir_operators[0].property.Mw['Solid']
            rho_a = rho_phases['aq']
            ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            kin_rates[i] = m.physics.reservoir_operators[0].property.kinetic_rate_ev.evaluate(kin_state, ss, rho_s,
                                                              m.physics.reservoir_operators[0].min_z,
                                                              m.physics.reservoir_operators[0].kin_fact)

            SR[i] = kin_state['SR']
            actHp[i] = kin_state['Act(H+)']

        for i in range(n_vars - 1):
            label = 'Operator Rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, op_kin_rates[:, i], color=colors[i], label=label)
            label = 'True rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates,
                       color=colors[i], linestyle='--', label=label)
            ax[0].plot(x[::n], m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates[::n],
                       color=colors[i], linestyle='None', marker='o', markerfacecolor='none', markersize=ms, label='_nolegend_')
        ax[1].plot(x, op_sr, color='b', label='Operator SR')
        ax[1].plot(x, SR, color='b', linestyle='--', label='True SR')
        ax[1].plot(x[::n], SR[::n], color='b', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')

        ax[2].plot(x, op_actHp, color='r', label=r'Operator $a_{H+}$')
        ax[2].plot(x, actHp, color='r', linestyle='--', label=r'True $a_{H+}$')
        ax[2].plot(x[::n], actHp[::n], color='r', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')
        # mult = SR
        # mult[SR > 100] = 100
        # ax[1].plot(x, 1 - mult, color='orange', linestyle='--', label=r'$1 - \hat{SR}$')

        ax[0].legend(loc='upper right', prop={'size': 12}, framealpha=0.9)
        ax[1].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[len(ax) - 1].set_xlabel('distance, m', fontsize=16)
        ax[0].set_ylabel('Kinetic rate, kmol / day / m3', fontsize=16)
        ax[1].set_ylabel('Saturation Ratio (SR)', fontsize=16)
        ax[2].set_ylabel('H+ ion activity', fontsize=16)
        fig.tight_layout()
        fig.savefig(f'1d_kin_rate_{t}.png', dpi=300)
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

def plot_max_cfl(paths, labels, nx, linestyle, colors):

    def extract_data(path):
        times = []
        time_steps = []
        cfl_numbers = []

        # Regular expression pattern to match the lines with T, DT, and CFL values
        pattern = re.compile(r"T = ([0-9.e+-]+), DT = ([0-9.e+-]+).*?CFL=([0-9.e+-]+)")

        # Open and process the log file
        with open(path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
                    time_steps.append(float(match.group(2)))
                    cfl_numbers.append(float(match.group(3)))
        return times, time_steps, cfl_numbers

    # colors = ['b', 'r', 'g', 'm', 'cyan']
    lw = 1
    ls = 12
    fs = 16

    fig, ax = plt.subplots(nrows=1, figsize=(8, 6))

    for i, path in enumerate(paths):
        time, steps, cfl = extract_data(path)
        time = np.array(time)
        steps = np.array(steps)
        cfl = np.array(cfl)
        ax.semilogy(time * 24, cfl, color=colors[i], lw=lw, ls=linestyle[i], label=labels[i])

    ax.set_xlabel('time, hours', fontsize=fs)
    ax.set_ylabel('Max CFL', fontsize=fs)
    ax.legend(loc='lower right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'cfl_cmp.png', dpi=300)
    # plt.show()
    plt.close(fig)

# paths = ['./100x100/data_ts3.vts',
#          './100x100/data_ts14.vts']
# write_2d_output_for_paper(paths=paths)

# 1D
# run(domain='1D', nx=100, max_ts=1.e-2)
run(domain='1D', nx=200, max_ts=1.e-3) # 4.e-5)
# run(domain='1D', nx=500, max_ts=5.e-4)

# paths = ['output_200/log.txt', 'output_2000_50000/log.txt',
#          'output_200_1000_5/log.txt', 'output_500/log.txt',
#          'output_200_1000_no_reaction/log.txt']
# labels = [r'$n_x=200,\, \Delta t_{max}=10^{-3}$ day, $n_{obl}=5001$',
#           r'$n_x=200,\, \Delta t_{max}=1\cdot 10^{-4}$ day, $n_{obl}=50001$',
#           r'$n_x=200,\, \Delta t_{max}=5\cdot 10^{-3}$ day, $n_{obl}=1001$',
#           r'$n_x=500,\, \Delta t_{max}=5\cdot 10^{-4}$ day \, $n_{obl}=5001$',
#           r'$n_x=200,\, \Delta t_{max}=8\cdot 10^{-6}$ day, $n_{obl}=5001$, no reaction']
# linestyle = ['-', '-.', ':', '-', '--']
# colors=['b', 'b', 'b', 'r', 'b']
# nx = [200, 200, 200, 500, 200]
# plot_max_cfl(paths=paths, labels=labels, nx=nx, linestyle=linestyle, colors=colors)

# 2D
# run(domain='2D', nx=10, max_ts=1.e-2)