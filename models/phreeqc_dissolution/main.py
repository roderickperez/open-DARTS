import os
os.environ["OMP_NUM_THREADS"] = "1"
from model import Model
from darts.engines import redirect_darts_output
from timestepping import DQNAgent
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

def run(self, days: float = None, restart_dt: float = 0., save_well_data: bool = True, save_solution_data: bool = True,
        log_3d_body_path: bool = False, verbose: bool = True):
    """
    Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

    :param days: Time increment [days]
    :type days: float
    :param restart_dt: Restart value for timestep size [days, optional]
    :type restart_dt: float
    :param verbose: Switch for verbose, default is True
    :type verbose: bool
    :param save_well_data: if True save states of well blocks at every time step to 'well_data.h5', default is True
    :type save_well_data: bool
    :param save_solution_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
    :type save_solution_data: bool
    :param log_3d_body_path: hypercube output
    :type verbose: bool
    """
    days = days if days is not None else self.runtime
    # get current engine time
    t = self.physics.engine.t
    stop_time = t + days

    # same logic as in engine.run
    if np.fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
        dt = self.params.first_ts
    elif restart_dt > 0.:
        dt = restart_dt
    else:
        dt = min(self.prev_dt * self.params.mult_ts, days, self.params.max_ts)
    self.prev_dt = dt

    ts = 0

    if log_3d_body_path:
        self.physics.body_path_start(output_folder=self.output_folder)

    # define positive-outcome and negative-outcome rewards
    def get_reward(outcome, prev_dt, dt, LI):
        a_ts = 100
        a_ni = 1
        return outcome * a_ts * np.log(dt / prev_dt) - a_ni * LI

    # initial agent state
    prev_agent_state = np.array([0.0, 0, 0, 0, 0.0, 0.0])
    action = self.agent.act(prev_agent_state)

    while t < stop_time:
        converged = self.run_timestep(dt, t, verbose)

        # assemble agent state
        max_residual = np.max(self.max_residual)
        nonlinear_iters = self.physics.engine.n_newton_last_dt
        linear_iters = self.physics.engine.n_linear_last_dt
        max_cfl = self.physics.engine.CFL_max
        dX = np.asarray(self.physics.engine.dX)
        X = np.asarray(self.physics.engine.X)
        max_change = np.where(X != 0, dX / X, 0).max()
        # TODO: extract time self.physics.engine.timer.get_timer()
        agent_state = np.array([max_residual, nonlinear_iters, linear_iters, max_cfl, max_change, dt])

        if converged:
            t += dt
            self.physics.engine.t = t
            ts += 1
            if verbose:
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                      % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

            # save state-response to the agent memory
            # reward = get_reward(outcome=1, prev_dt=self.prev_dt, dt=dt, LI=self.physics.engine.n_linear_last_dt)
            # self.agent.remember(state=prev_agent_state, action=action, reward=reward, next_state=agent_state, done=True)
            # self.agent.replay(self.agent_batch_size)

            # new action
            # action = self.agent.act(agent_state)
            # dt_factor = self.agent_actions_map[action]
            # dt = dt_factor * dt
            dt = min(dt * self.params.mult_ts, self.params.max_ts)

            # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
            # to not allow the next time step be smaller than min_ts
            if np.fabs(t + dt - stop_time) < self.params.min_ts:
                dt = stop_time - t

            if t + dt > stop_time:
                dt = stop_time - t
            else:
                self.prev_dt = dt

            if log_3d_body_path:
                self.physics.body_path_add_bodys(output_folder=self.output_folder, time=t)

            if save_well_data:
                self.save_data_to_h5(kind='well')

        else:
            # save state-response to the agent memory
            # reward = get_reward(outcome=-1, prev_dt=self.prev_dt, dt=dt, LI=self.physics.engine.n_linear_last_dt)
            # self.agent.remember(state=prev_agent_state, action=action, reward=reward, next_state=agent_state, done=True)
            # self.agent.replay(self.agent_batch_size)

            # new action
            # action = self.agent.act(agent_state)
            # dt_factor = self.agent_actions_map[action]
            # dt = dt_factor * dt

            dt /= self.params.mult_ts

            if verbose:
                print("Cut timestep to %2.10f" % dt)
            assert dt > self.params.min_ts, ('Stop simulation. Reason: reached min. timestep '
                                             + str(self.params.min_ts) + ' dt=' + str(dt))

        prev_agent_state = agent_state

    # update current engine time
    self.physics.engine.t = stop_time

    # save solution vector
    if save_solution_data:
        self.save_data_to_h5(kind='solution')

    if verbose:
        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
              % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                 self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                 self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
    return 0

def run_simulation(domain: str, max_ts: float, nx: int = 100, poro_filename: str = None, output: bool = False):
    # Make a folder
    output_folder = 'output_' + domain + '_' + str(nx)
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Redirect output to log file
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    # Create model
    m = Model(domain=domain, nx=nx, poro_filename=poro_filename)
    m.sol_filename = 'nx' + str(nx) + '.h5'

    # Initialize model
    m.init(output_folder=output_folder)
    m.physics.engine.n_solid = 1

    # Timestepping agent
    state_size = 6  # e.g., 6 features
    action_size = 3  # decrease, maintain, increase
    m.agent = DQNAgent(state_size=state_size, action_size=action_size, model_type='linear')
    m.agent_batch_size = 32
    m.agent_actions_map = {0: 0.5, 1: 1.0, 2: 2.0}

    # Initialization check
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]
    volume = np.array(m.reservoir.mesh.volume, copy=False)
    total_pv = np.sum(volume[:m.reservoir.n] * poro) * 1e6
    print(f'Total pore volume: {total_pv} cm3')
    print(f'Injection rate: {m.inj_cells.size} cells * {m.inj_rate / total_pv * 1e+6} PV/day')

    m.params.max_ts = max_ts

    ith_step = 0
    if domain == '1D':
        if output: plot_profiles(m)
        run(self=m, days=0.002, restart_dt=max_ts)
        if output: plot_profiles(m)
        run(self=m, days=0.018, restart_dt=max_ts)
        if output: plot_profiles(m)
        run(self=m, days=0.02, restart_dt=max_ts)
        if output: plot_profiles(m)
        m.params.max_ts *= 3
        m.params.first_ts = m.params.max_ts
        run(self=m, days=0.1, restart_dt=max_ts)
        m.params.max_ts *= 4
        run(self=m, days=0.86)
        if output: plot_profiles(m)
        m.params.max_ts *= 5
        m.params.first_ts = m.params.max_ts
    else:
        if output: m.output_to_vtk(ith_step=ith_step)
        ith_step += 1
        run(self=m, days=0.04, restart_dt=max_ts)
        if output: m.output_to_vtk(ith_step=ith_step)
        ith_step += 1
        m.params.max_ts *= 40
        m.params.first_ts = m.params.max_ts

    for i in range(2):
        if domain == '1D': dt = 7.0
        else: dt = 2.0
        run(self=m, days=dt)
        if i < 1:
            m.params.max_ts *= 1.5
            m.params.first_ts = m.params.max_ts
        if output:
            if domain == '1D':
                plot_profiles(m)
            else:
                m.output_to_vtk(ith_step=ith_step)
                ith_step += 1

    # Print some statistics
    print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')

    m.print_timers()
    m.print_stat()

def plot_profiles(m):
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    timesteps, property_array = m.output_properties(output_properties=props_names)

    # add porosity & hydrogen
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]
    property_array['porosity'] = poro[np.newaxis]

    # folder
    t = round(m.physics.engine.t, 4)
    path = os.path.join(m.output_folder, str(t))
    if not os.path.exists(path):
        os.makedirs(path)

    ls = 14
    x = m.reservoir.discretizer.centroids_all_cells[:, 0]
    for prop, data in property_array.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, data[0, :], color='b', label=prop)
        ax.set_xlabel('distance, m', fontsize=16)
        t_hours = round(m.physics.engine.t * 24, 4)
        ax.text(0.21, 0.85, 'time = ' + str(t_hours) + ' hours', fontsize=16, rotation='horizontal', transform=fig.transFigure)

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


def plot_old_profiles(m, output_folder, plot_kinetics=False):
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

if __name__ == '__main__':
    # 1D
    run_simulation(domain='1D', nx=200, max_ts=1.e-3)  # 4.e-5)
    # run_simulation(domain='1D', nx=500, max_ts=5.e-4)

    # 2D
    # run_simulation(domain='2D', nx=10, max_ts=2.e-3)
    # run_simulation(domain='2D', nx=100, max_ts=5.e-4, poro_filename='spherical_100_8.txt')

