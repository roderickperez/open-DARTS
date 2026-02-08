from model import Model
from darts.engines import *
import numpy as np
import meshio
from math import fabs

def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.params.mult_ts
    max_dt = m.params.max_ts
    m.e = m.physics.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if fabs(t) < 1e-15:
        dt = m.params.first_ts
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.params.max_ts

    # evaluate end time
    runtime += t
    ts = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            # evaluate and assign transient boundaries or sources / sinks
            # m.reservoir.update_well(dt=dt, time=new_time, physics=m.physics)
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t)

        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                   % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))

            dt *= mult_dt
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        else:
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %2.3f" % dt)
            if dt < 1e-8:
               break
    # update current engine time
    m.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (m.e.stat.n_timesteps_total, m.e.stat.n_timesteps_wasted,
                                                        m.e.stat.n_newton_total, m.e.stat.n_newton_wasted,
                                                        m.e.stat.n_linear_total, m.e.stat.n_linear_wasted))
def run_timestep_python(m, dt, t):
    self = m
    max_newt = self.params.max_i_newton
    self.e.n_linear_last_dt = 0
    well_tolerance_coefficient = 1e2
    self.timer.node['simulation'].start()
    for i in range(max_newt + 1):
        self.e.run_single_newton_iteration(dt)
        res = self.e.calc_newton_dev()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        if len(res) > 2 and res[2] == res[2]:       self.e.dev_g = res[2]
        else:                                       self.e.dev_g = 0.0

        self.e.newton_residual_last_dt = 0.0#np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + self.e.dev_g ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'res_p = ' + str(self.e.dev_p) + '\t' + 'res_u = ' + str(self.e.dev_u) + '\t' + \
                    'res_g = ' + str(self.e.dev_g) + '\t' + 'dev_well = ' + str(self.e.well_residual_last_dt))
        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and self.e.dev_u < self.params.tolerance_newton and self.e.dev_g < self.params.tolerance_newton
           and self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
              or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                break
        r_code = self.e.solve_linear_equation()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
    # End of newton loop
    converged = self.e.post_newtonloop(dt, t)
    self.timer.node['simulation'].stop()
    return converged

def just_run():
    #t0 = 0.000125
    nt = 30
    max_t = 3.0
    t = np.logspace(-5, np.log10(max_t), nt)
    #t = t0 * np.ones(nt)
    m = Model()
    m.init()
    redirect_darts_output('log.txt')
    output_directory = 'sol_{:s}'.format(m.physics_type)
    m.timer.node["update"] = timer_node()
    ith_step = 0  # Store initial conditions as ../solution0.vtk
    m.physics.engine.print_linear_system = False

    #X = np.array(m.physics.engine.X, copy=False)
    # find equilibrium
    # m.reservoir.set_equilibrium()
    # m.physics.engine.find_equilibrium = True
    # m.params.first_ts = 1
    # run_python(m, 1.0, init_step=True)
    # m.reinit(output_directory)

    m.physics.engine.find_equilibrium = False
    m.physics.engine.contact_solver = contact_solver.RETURN_MAPPING
    m.setup_contact_friction(m.physics.engine.contact_solver)
    if m.params.linear_type == sim_params.cpu_gmres_fs_cpr:
        m.physics.engine.update_uu_jacobian()

    m.physics.engine.t = 0.0
    restart_dt = -1
    time = 0
    for ith_step, dt in enumerate(t):
        time += dt
        m.params.first_ts = dt
        m.params.max_ts = dt
        run_python(m, dt, restart_dt)
        m.reservoir.write_to_vtk(output_directory, ith_step + 1, m.physics)
    m.print_timers()

if __name__ == '__main__':
    pass
    #just_run()