import os
# os.environ["OMP_NUM_THREADS"] = "1"
from model import Model
from darts.engines import redirect_darts_output
import numpy as np
from visualization import plot_profiles, animate_1d

def run_simulation(domain: str, max_ts: float, nx: int = 100, mesh_filename: str = None, poro_filename: str = None, output: bool = False):
    # Make a folder
    output_folder = 'output_' + domain + '_' + str(nx)
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Redirect output to log file
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    # Create model
    m = Model(domain=domain, nx=nx, mesh_filename=mesh_filename, poro_filename=poro_filename)
    m.sol_filename = 'nx' + str(nx) + '.h5'

    # Initialize model
    m.init(output_folder=output_folder)
    m.physics.engine.n_solid = 1

    # Timestepping agent
    # state_size = 6  # e.g., 6 features
    # action_size = 3  # decrease, maintain, increase
    # m.agent = DQNAgent(state_size=state_size, action_size=action_size, model_type='linear')
    # m.agent_batch_size = 32
    # m.agent_actions_map = {0: 0.5, 1: 1.0, 2: 2.0}

    # Initialization check
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]
    volume = np.array(m.reservoir.mesh.volume, copy=False)
    total_pv = np.sum(volume[:m.n_res_blocks] * poro) * 1e6
    print(f'Total pore volume: {total_pv} cm3')
    print(f'Injection rate: {m.inj_cells.size} cells * {m.inj_rate / total_pv * 1e+6} PV/day')

    # visualization
    def plot(m, ith_step=None):
        if output:
            if domain == '1D': return plot_profiles(m, output_folder=output_folder)
            else: m.output_to_vtk(ith_step=ith_step)

    m.params.max_ts = max_ts
    ith_step = 0
    if domain == '1D':
        fig_paths = []
        fig_paths.append(plot(m))
        m.run(days=0.002, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.run(days=0.018, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.run(days=0.02, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.params.max_ts *= 3
        m.params.first_ts = m.params.max_ts
        m.run(days=0.1, restart_dt=max_ts)
        m.params.max_ts *= 4
        m.run(days=0.86)
        fig_paths.append(plot(m))
        m.params.max_ts *= 5
        m.params.first_ts = m.params.max_ts

        for i in range(7):
            dt = 2.0
            m.run(days=dt)
            if i < 1:
                m.params.max_ts *= 1.5
                m.params.first_ts = m.params.max_ts
            fig_paths.append(plot(m))

            if output:
                animate_1d(output_folder=output_folder, fig_paths=fig_paths)
    elif domain == '2D':
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001, restart_dt=max_ts)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.params.max_ts *= 40
        m.params.first_ts = m.params.max_ts
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1

        for i in range(15):
            dt = 0.4
            m.run(days=dt)
            if i < 1:
                m.params.max_ts *= 1.5
                m.params.first_ts = m.params.max_ts
            plot(m=m, ith_step=ith_step)
            ith_step += 1
    elif domain == '3D':
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001, restart_dt=max_ts)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.params.max_ts *= 20
        m.params.first_ts = m.params.max_ts
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1

        for i in range(15):
            dt = 0.4
            m.run(days=dt)
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
    # run_simulation(domain='2D', nx=100, max_ts=8.e-4, output=True, poro_filename='input/spherical_100_8.txt')

    # 3D
    # run_simulation(domain='3D', max_ts=2.e-3, output=False,
    #                mesh_filename='input/core_13k.msh', poro_filename='input/core_13k_0.02.txt')
    # run_simulation(domain='3D', max_ts=1.e-3, output=True,
    #                mesh_filename='input/core_60k.msh', poro_filename='input/core_60k_0.01.txt')
    # run_simulation(domain='3D', max_ts=8.e-4, output=True,
    #                mesh_filename='input/core_195k.msh', poro_filename='input/core_195k.txt')


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
