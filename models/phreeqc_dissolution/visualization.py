import pandas as pd
import numpy as np
import re
import subprocess
import os
import matplotlib
matplotlib.use('pgf')
matplotlib.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{color}')
from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=16)

def plot_new_profiles(m):
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    timesteps, property_array = m.output_properties(output_properties=props_names)

    # folder
    t = round(m.physics.engine.t, 4)
    path = os.path.join(m.output_folder, str(t))
    if not os.path.exists(path):
        os.makedirs(path)

    ls = 18
    x = m.reservoir.discretizer.centroids_all_cells[:, 0]
    for prop, data in property_array.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, data[0, :], color='b', label=prop)
        ax.set_xlabel('distance, m', fontsize=16)
        t_hours = round(m.physics.engine.t * 24, 4)
        ax.text(0.71, 0.85, 'time = ' + str(t_hours) + ' hours', fontsize=16, rotation='horizontal', transform=fig.transFigure)

        # y-axis limits
        if prop == 'p':
            ax.set_ylabel('pressure, bar', fontsize=16)
            ax.set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.1)
        elif prop == 'porosity':
            ax.set_ylabel('porosity', fontsize=16)
            ax.set_ylim(-0.01, 1.01)
        else:
            ax.set_ylabel(prop, fontsize=16)
            # fig.gca().set_ylim(bottom=-0.0001)

        fig.tight_layout()
        fig_name = os.path.join(path, f'{prop}.png')
        fig.savefig(fig_name, dpi=300)
        # plt.show()
        plt.close(fig)

def plot_current_profiles(m, output_folder='./', plot_kinetics=False):
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
    ax[1].plot(x, 1.0 - np.sum(Xm[:, 2:], axis=1), color=colors[n_vars - 1], label=m.physics.property_containers[0].components_name[-1])
    ax[2].plot(x, poro, color=colors[0], label='porosity')

    t = round(m.physics.engine.t * 24, 4)
    ax[0].text(0.21, 0.9, 'time = ' + str(t) + ' hours',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[n_plots - 1].set_xlabel('distance, m', fontsize=16)
    ax[1].set_ylabel(r'\textcolor{blue}{z$_{CaCO3}$(s)}, \textcolor{magenta}{z$_O$} and \textcolor{cyan}{z$_H$}', fontsize=20)
    ax[2].set_ylabel('porosity', fontsize=16)
    ax1.set_ylabel(r'\textcolor{red}{z$_{Ca}$} and \textcolor{green}{z$_C$}', fontsize=20)

    ax[0].set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.3)
    ax[1].set_ylim(-0.01, 1.01)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.01)
    ls = 14
    ax[0].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()
    fig_name = os.path.join(output_folder, f'1d_dissolution_time_{round(m.physics.engine.t, 4)}.png')
    fig.savefig(fig_name, dpi=300)
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
        fig.savefig(os.path.join(output_folder, f'1d_kin_rate_{t}.png'), dpi=300)
        plt.close(fig)

def plot_profiles(m, output_folder='./', plot_kinetics=False, plot_saturation=True):
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]

    # fig = plt.figure(figsize=(16, 6))
    # # Create a 1x3 grid with different column widths (adjust as needed):
    # gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[0.3, 0.4, 0.3])#, wspace=0.2)
    # ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    n_plots = 3
    fig, ax = plt.subplots(ncols=n_plots, sharex=True, figsize=(16, 6))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]

    ax[0].plot(x, Xm[:, 0], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = ['b', 'r', 'g', 'm', 'cyan']

    for i in [1,4]: # Solid / O
        label = r'$\mathrm{CaCO_3}$(s)' if i == 1 else m.physics.vars[i]
        ax[1].plot(x, Xm[:, i], color=colors[i-1], label=label)
    for i in [2,3]: # Ca / C
        ax1.plot(x, Xm[:, i], color=colors[i-1], label=m.physics.vars[i])
    ax[1].plot(x, 1.0 - np.sum(Xm[:, 2:], axis=1), color=colors[n_vars - 1], label=m.physics.property_containers[0].components_name[-1])
    ax[2].plot(x, poro, color=colors[0], label='porosity')

    t = round(m.physics.engine.t * 24, 4)
    ax[0].text(0.15, 0.85, 'time = ' + str(t) + ' hours',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[0].set_xlabel('distance, m', fontsize=16)
    ax[1].set_xlabel('distance, m', fontsize=16)
    ax[2].set_xlabel('distance, m', fontsize=16)
    ax[1].set_ylabel(r'\textcolor{blue}{z$_{CaCO3}$(s)}, \textcolor{magenta}{z$_O$} and \textcolor{cyan}{z$_H$}', fontsize=16)
    ax[2].set_ylabel('porosity', fontsize=16)
    ax1.set_ylabel(r'\textcolor{red}{z$_{Ca}$} and \textcolor{green}{z$_C$}', fontsize=20)

    ax[0].set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.15)
    ax[1].set_ylim(-0.01, 1.01)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.01)
    ls = 14
    ax[0].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax[2].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    if plot_saturation:
        ax_sat = ax[2].twinx()
        gas_sat = np.zeros(n_cells)
        prop = m.physics.reservoir_operators[0].property
        for i in range(n_cells):
            nu_v, _, _, rho_phases, _, _, _ = prop.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1]
            nu_v = nu_v * (1 - nu_s)  # convert to overall molar fraction
            nu_a = 1 - nu_v - nu_s
            rho_a, rho_v = rho_phases['aq'], rho_phases['gas']
            rho_s = prop.density_ev['solid'].evaluate(Xm[i, 0]) / prop.Mw['Solid']
            if nu_v > 0:
                sv = nu_v / rho_v / (nu_v / rho_v + nu_a / rho_a + nu_s / rho_s)
                sa = nu_a / rho_a / (nu_v / rho_v + nu_a / rho_a + nu_s / rho_s)
                ss = nu_s / rho_s / (nu_v / rho_v + nu_a / rho_a + nu_s / rho_s)
            else:
                sv = 0
                sa = nu_a / rho_a / (nu_a / rho_a + nu_s / rho_s)
                ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            gas_sat[i] = sv
        ax_sat.plot(x, gas_sat, color=colors[1], label='gas saturation')
        ax_sat.set_ylabel('gas saturation', fontsize=16)
        ax_sat.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()

    pos0 = ax[0].get_position()  # [x0, y0, width, height]
    pos1 = ax[1].get_position()
    pos2 = ax[2].get_position()
    ax[0].set_position([pos0.x0 + 0.05, pos0.y0, pos0.width, pos0.height])
    fig_name = os.path.join(output_folder, f'1d_dissolution_time_{round(m.physics.engine.t, 4)}.png')
    fig.savefig(fig_name, dpi=300)
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
        fig.savefig(os.path.join(output_folder, f'1d_kin_rate_{t}.png'), dpi=300)
        plt.close(fig)

    return fig_name

def animate_1d(output_folder, fig_paths):
    # Rename them to a zero-padded sequence: frame_0000.png, frame_0001.png, etc.
    # so ffmpeg can pick them up in order.
    for i, old_name in enumerate(fig_paths):
        new_name = os.path.join(output_folder, f"frame_{i:04d}.png")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)

    # Now call ffmpeg to produce a video "simulation_animation.mp4"
    fps = 1
    ffmpeg_path = r'c:\work\packages\ffmpeg-6.0\bin\ffmpeg.exe'
    output_video = os.path.join(output_folder, "simulation_animation.mp4")
    cmd = [
        ffmpeg_path,
        '-y',
        '-framerate', str(fps),  # frames per second
        '-i', os.path.join(output_folder, 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
        '-c:v', 'libx264',  # H.264 encoding
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-movflags', '+faststart',
        output_video
    ]
    subprocess.run(cmd, check=True)

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


