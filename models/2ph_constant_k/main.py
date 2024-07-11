import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

from model import Model
from darts.engines import redirect_darts_output
from darts.physics.operators_base import PropertyOperators as props
from darts.tools.hdf5_tools import load_hdf5_to_dict

rcParams["text.usetex"]=False
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

def find_ffmpeg():
    common_paths = [
        '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg',  # Unix-like systems
        'C:\\ffmpeg\\bin\\ffmpeg.exe',               # Windows
        '/opt/local/bin/ffmpeg',                     # Other potential locations
        'C:\\work\\packages\\ffmpeg-6.0\\bin\\ffmpeg.exe'
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return os.getenv('FFMPEG_PATH', 'ffmpeg')  # Fallback to hoping it's in the PATH

rcParams['animation.ffmpeg_path'] = find_ffmpeg()

def animate_solution_1d(paths, n_cells, labels, lower_lims, upper_lims):
    data0 = load_hdf5_to_dict(filename=paths[0] + 'solution.h5')['dynamic']
    c = np.arange(n_cells[0])
    n_plots = len(data0['variable_names'])
    n_steps = data0['time'].size # number of saved snapshots
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(8, 14))
    for i in range(n_plots):
        if data0['variable_names'][i] == 'pressure':
            ax[i].set_ylabel('pressure, bar', fontsize=14)
        else:
            ax[i].set_ylabel(data0['variable_names'][i], fontsize=14)
    ax[n_plots - 1].set_xlabel('x, m', fontsize=16)

    lines = []
    i = 0
    for k, path in enumerate(paths):
        filename = path + 'solution.h5'
        data = load_hdf5_to_dict(filename=filename)['dynamic']

        for i in range(n_plots):
            li, = ax[i].plot(c, data['X'][0,:n_cells[k],i], linewidth=1, color=colors[k], linestyle='-', label=labels[k])
            lines.append(li)

    time_text = ax[0].text(0.4, 0.93, 'time = ' + str(round(data0['time'][0], 4)) + ' sec',
                           fontsize=16, rotation='horizontal', transform=fig.transFigure)

    for i in range(n_plots):
        ax[i].set_ylim(lower_lims[i], upper_lims[i])

    ax[0].legend(loc='upper right', prop={'size': 14})
    nt = n_steps # num of updates

    def animate(i):
        time_to_update = False
        for k, path in enumerate(paths):
            each_ith = int(n_steps / nt)
            ind = int(n_steps * i / n_steps)
            if ind % each_ith == 0:
                data = load_hdf5_to_dict(filename=path + 'solution.h5')['dynamic']
                for j in range(n_plots):
                    lines[n_plots * k + j].set_data(c, data['X'][i,:n_cells[k],j])
                time_text.set_text('time = ' + str(round(data0['time'][i], 4)) + ' sec')

        return lines


    anim = FuncAnimation(fig, animate, interval=100, repeat=True, frames=np.arange(0, n_steps))

    # writer = animation.PillowWriter(fps=20,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # anim.save(paths[0] + 'comparison.gif', writer=writer)

    fig.tight_layout()
    writervideo = animation.FFMpegWriter(fps=1)
    anim.save(paths[0] + 'comparison.mp4', writer=writervideo)
    plt.close(fig)

def run(itor_mode, itor_type, obl_points, reservoir_type, nx: int = None):
    output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_' + reservoir_type
    if itor_type == 'linear':
        log_3d_body_path = False
    else:
        log_3d_body_path = True

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    components = ['CO2', 'CH4', 'C4', 'C10', 'C16', 'C20']
    inj_comp = [0.995, 0.001, 0.001, 0.001, 0.001, 0.001]
    ini_comp = [0.005, 0.350, 0.250, 0.195, 0.125, 0.075]

    redirect_darts_output('log.out')
    n = Model(inj_comp=inj_comp, ini_comp=ini_comp, obl_points=obl_points,
              reservoir_type=reservoir_type, nx=nx)
    n.init(itor_mode=itor_mode, itor_type=itor_type, output_folder=output_folder)

    if reservoir_type != '1D':
        n.output_to_vtk(ith_step=0)
        if reservoir_type != '2D':
            n.params.first_ts = n.params.max_ts = 0.001
            n.run(1.0, log_3d_body_path=log_3d_body_path)
            n.physics.engine.t = 0.0
            n.set_spe10_well_controls_initialized()


    for i in range(12):
        n.run(30.5, log_3d_body_path=log_3d_body_path)
        if reservoir_type != '1D':
            n.output_to_vtk(ith_step=i + 1)

    n.print_timers()
    n.print_stat()

    if reservoir_type == '1D':
        # populate input lists for comparing multiple solutions
        animate_solution_1d(paths=[output_folder + '/'],
                            labels=[itor_type + ', ' + itor_mode + ', N=' + str(nx)],
                            n_cells=[nx],
                            lower_lims = [48.9, -1.e-2, -1.e-2, -1.e-2, -1.e-2, -1.e-2],
                            upper_lims = [60, 1.01, 0.5, 0.21, 0.15, 0.1])

# run(itor_type='linear', itor_mode='adaptive', obl_points=1024, reservoir_type='1D', nx=100)
run(itor_type='linear', itor_mode='adaptive', obl_points=1024, reservoir_type='2D', nx=100)
# run(itor_type='linear', itor_mode='adaptive', obl_points=1024, reservoir_type='spe10_20_40_40')
# run(itor_type='multilinear', itor_mode='adaptive', obl_points=1024)
# run(itor_mode='static_nested', obl_points=4)