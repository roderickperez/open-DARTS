import os
# os.environ["OMP_NUM_THREADS"] = "1"
from model import Model
from darts.engines import redirect_darts_output
from timestepping import DQNAgent
import numpy as np

from visualization import plot_profiles, animate_1d

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

    # visualization
    def plot(m, ith_step=None):
        if output:
            if domain == '1D': return plot_profiles(m, output_folder=output_folder)
            elif domain == '2D': m.output_to_vtk(ith_step=ith_step)

    m.params.max_ts = max_ts
    ith_step = 0
    if domain == '1D':
        fig_paths = []
        fig_paths.append(plot(m))
        run(self=m, days=0.002, restart_dt=max_ts)
        fig_paths.append(plot(m))
        run(self=m, days=0.018, restart_dt=max_ts)
        fig_paths.append(plot(m))
        run(self=m, days=0.02, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.params.max_ts *= 3
        m.params.first_ts = m.params.max_ts
        run(self=m, days=0.1, restart_dt=max_ts)
        m.params.max_ts *= 4
        run(self=m, days=0.86)
        fig_paths.append(plot(m))
        m.params.max_ts *= 5
        m.params.first_ts = m.params.max_ts

        for i in range(7):
            dt = 2.0
            run(self=m, days=dt)
            if i < 1:
                m.params.max_ts *= 1.5
                m.params.first_ts = m.params.max_ts
            fig_paths.append(plot(m))

            if output:
            animate_1d(output_folder=output_folder, fig_paths=fig_paths)
    elif domain == '2D':
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        run(self=m, days=0.001, restart_dt=max_ts)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.params.max_ts *= 40
        m.params.first_ts = m.params.max_ts
        run(self=m, days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        run(self=m, days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1

        for i in range(15):
            dt = 0.4
        run(self=m, days=dt)
        if i < 1:
            m.params.max_ts *= 1.5
            m.params.first_ts = m.params.max_ts
            plot(m=m, ith_step=ith_step)
                ith_step += 1

    # Print some statistics
    print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')

    m.print_timers()
    m.print_stat()

if __name__ == '__main__':
    # 1D
    run_simulation(domain='1D', nx=200, max_ts=1.e-3)  # 4.e-5)
    # run_simulation(domain='1D', nx=500, max_ts=5.e-4)

    # 2D
    # run_simulation(domain='2D', nx=10, max_ts=2.e-3)
    # run_simulation(domain='2D', nx=100, max_ts=8.e-4, output=True, poro_filename='spherical_100_8.txt')

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
