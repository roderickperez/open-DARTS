from model import Model
from darts.engines import *
import numpy as np
import meshio
from math import fabs

def run_python(m, days=0, restart_dt=0, init_step = False):
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
            m.reservoir.update_core_2d_boundary(dt=dt, time=new_time, physics=m.physics)
            #if res == -1:
            #    return -1
                #new_time -= dt
                #dt = m.reservoir.max_dt_change
                #continue
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t, init_step)

        m.reservoir.mech_operators.eval_stresses(m.physics.engine.fluxes, m.physics.engine.fluxes_biot, m.physics.engine.X,
                                                 m.reservoir.mesh.bc, m.physics.engine.op_vals_arr)

        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d" % (ts, t * 86400, dt * 86400, m.e.n_newton_last_dt, m.e.n_linear_last_dt))

            if init_step:
                m.e.dt1 = 0.0

            dt *= mult_dt
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        else:
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %f sec" % (dt * 86400.0))
    # update current engine time
    m.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (m.e.stat.n_timesteps_total, m.e.stat.n_timesteps_wasted,
                                                        m.e.stat.n_newton_total, m.e.stat.n_newton_wasted,
                                                        m.e.stat.n_linear_total, m.e.stat.n_linear_wasted))
def run_python_fit_bc(m, days=0, restart_dt=0, init_step = False):
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

        m.timer.node["update"].start()
        # store boundaries taken at previous time step
        m.reservoir.update(dt=dt, time=new_time)
        # evaluate and assign transient boundaries or sources / sinks
        m.reservoir.update_core_2d_boundary(dt=dt, time=new_time, physics=m.physics)
        # update transient boundaries or sources / sinks
        m.reservoir.update_trans(dt, m.physics.engine.X)
        m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t)

        m.reservoir.mech_operators.eval_stresses(m.physics.engine.fluxes, m.physics.engine.fluxes_biot, m.physics.engine.X,
                                                 m.reservoir.mesh.bc, m.physics.engine.op_vals_arr)

        rel_tol = 0.03
        resulted_s1 = -np.array(m.reservoir.mech_operators.total_stresses, copy=False)[6 * m.reservoir.ref_cell_id + 1]
        prescribed_s1 = 10 * np.interp(new_time * 86400, m.reservoir.time_load, m.reservoir.stress_to_set)
        if new_time > 50 / 86400.0:
            ds1 = prescribed_s1 - resulted_s1
        else:
            ds1 = 0.0

        if converged and np.fabs(ds1 / prescribed_s1) < rel_tol:
            print("Rel. diff. in s1 = %f, un_top = %f at t = %f" % (np.fabs(ds1 / prescribed_s1), m.reservoir.un_top, t * 86400.0))
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d" % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))

            dt *= mult_dt
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        elif converged and np.fabs(ds1 / prescribed_s1) >= rel_tol:
            print("Rel. diff. in s1 = %f, un_top = %f at t = %f" % (ds1 / prescribed_s1, m.reservoir.un_top, new_time * 86400.0))
            m.reservoir.un_top -= ds1 / m.reservoir.E * m.dims[1]
        else:
            dt /= mult_dt
            print("Cut timestep to %f sec" % (dt * 86400.0))
    # update current engine time
    m.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (m.e.stat.n_timesteps_total, m.e.stat.n_timesteps_wasted,
                                                        m.e.stat.n_newton_total, m.e.stat.n_newton_wasted,
                                                        m.e.stat.n_linear_total, m.e.stat.n_linear_wasted))
def run_timestep_python(m, dt, t, init_step):
    self = m
    max_newt = self.params.max_i_newton
    self.e.n_linear_last_dt = 0
    well_tolerance_coefficient = 1e2
    self.timer.node['simulation'].start()

    # if not init_step and t > 0.0:
    #     if self.e.dt1 > 0:
    #         deltaX = (np.array(self.e.Xn, copy=False) - np.array(self.e.Xn1, copy=False)) / self.e.dt1
    #     else:
    #         deltaX = (np.array(self.e.Xn, copy=False) - np.array(self.e.Xn1, copy=False))
    #     X = np.array(self.e.X, copy=False)
    #     for contact in self.physics.engine.contacts:
    #         cell_ids = np.array(self.e.contacts[0].cell_ids, dtype=np.intc)[np.where(np.array(contact.states) == contact_state.SLIP)[0]]
    #         for i in range(3):
    #             X[4 * cell_ids + contact.U_VAR + i] += deltaX[4 * cell_ids + contact.U_VAR + i] * dt / 5.0

    np.array(self.e.dX, copy=False)[:] = 0.0
    res_history = []
    for i in range(max_newt+1):
        self.e.run_single_newton_iteration(dt)
        res = self.e.calc_newton_dev()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        self.e.dev_g = res[2]
        res_history.append((res[0], res[1], res[2]))

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2)
        dg_norm = 0#np.linalg.norm(np.array(self.e.dX[-4*self.reservoir.unstr_discr.frac_cells_tot:]).reshape(self.reservoir.unstr_discr.frac_cells_tot, 4)[:,:3], axis=1)
        #print(dg_norm)
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'res_p = ' + str(res[0]) + '\t' + 'res_u = ' + str(res[1]) + '\t' + \
                    'res_g = ' + str(res[2]))
        #for ith_contact, contact in enumerate(self.physics.engine.contacts):
        #    print('fault #' + str(ith_contact) + ': ' + str(contact.num_of_change_sign))
        self.e.n_newton_last_dt = i

        #self.reservoir.write_to_vtk('sol_{:s}'.format(m.physics_type), i + 1, m.physics)

        #  check tolerance if it converges
        if ((self.e.dev_u / res_history[0][1] < self.params.tolerance_newton / 100 and
             self.e.dev_g / res_history[0][2] < self.params.tolerance_newton and
             self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
                or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                break

        # line search
        if False and i > 1 and res_history[-2][0] < self.params.tolerance_newton and res_history[-1][1] > 0.9 * res_history[-2][1]:
            coef = np.array([0.0, 1.0])
            history = np.array([res_history[-2], res_history[-1]])
            linear_search(self, dt, coef, history)
        else:
            r_code = self.e.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.e.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
    # End of newton loop
    converged = self.e.post_newtonloop(dt, t, 1)
    self.timer.node['simulation'].stop()
    return converged
def linear_search(self, dt, coef, history):
        print('LS: ' + str(coef[0]) + '\t' + 'res_p = ' + str(history[0][0]) + '\tres_u = ' + str(history[0][1]))
        print('LS: ' + str(coef[1]) + '\t' + 'res_p = ' + str(history[1][0]) + '\tres_u = ' + str(history[1][1]))
        res_history = np.array([history[0][1], history[1][1]])

        for iter in range(5):
            if coef.size > 2:
                id = res_history.argmin()
                closest_left = np.where(coef < coef[id])[0]
                closest_right = np.where(coef > coef[id])[0]
                if closest_left.size and closest_right.size:
                    left = closest_left[coef[closest_left].argmax()]
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[left] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[left]) / 2)
                    elif res_history[right] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[right]) / 2)
                    else:
                        if res_history[left] < res_history[right]:
                            coef = np.append(coef, coef[id] - (coef[id] - coef[left]) / 4)
                        else:
                            coef = np.append(coef, coef[id] + (coef[right] - coef[id]) / 4)
                elif closest_left.size:
                    left = closest_left[coef[closest_left].argmax()]
                    if res_history[left] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[left]) / 2)
                    else:
                        coef = np.append(coef, coef[id] + (coef[id] - coef[left]) / 2)
                elif closest_right.size:
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[right] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[right]) / 2)
                    else:
                        coef = np.append(coef, coef[id] - (coef[right] - coef[id]) / 2)
                if coef[-1] <= 0: coef[-1] = 1.E-2
                if coef[-1] >= 1: coef[-1] = 1.0 - 1.E-2
            else:
                coef = np.append(coef, coef[-1] / 2)

            self.e.newton_update_coefficient = coef[-1] - coef[-2]
            self.e.apply_newton_update(dt)
            self.e.run_single_newton_iteration(dt)
            res = self.e.calc_newton_dev()
            res_history = np.append(res_history, res[1])
            print('LS: ' + str(coef[-1]) + '\t' + 'res_p = ' + str(res[0]) + '\tres_u = ' + str(res[1]))

        final_id = res_history.argmin()
        self.e.newton_update_coefficient = coef[final_id] - coef[-1]
        self.e.apply_newton_update(dt)

def just_run():
    # sec
    #t0 = 20.0 / 86400
    #nt = 200
    max_t = 1900.0 / 86400.0
    dt = 100.0 / 86400.0
    #t = np.logspace(-1, np.log10(max_t), nt)
    #t = t0 * np.ones(nt)
    m = Model()
    m.init()
    redirect_darts_output('log.txt')
    output_directory = 'sol_{:s}'.format(m.physics_type)
    m.timer.node["update"] = timer_node()

    ith_step = 0
    # set equilibrium (including boundary conditions)
    # m.reservoir.set_equilibrium()
    # m.physics.engine.find_equilibrium = True
    # m.params.first_ts = 1
    # run_python(m, 1.0)
    # m.reinit_reference(output_directory)

    # m.physics.engine.x_dim = 0.001
    # m.physics.engine.p_dim = 1.0
    # m.physics.engine.t_dim = 0.01
    # m.physics.engine.print_linear_system = True
    m.physics.engine.scale_rows = True
    # m.reservoir.set_equilibrium()
    # m.physics.engine.find_equilibrium = True
    # m.params.first_ts = 1
    # run_python(m, 1.0, init_step=True)
    # m.reinit(output_directory)

    #m.params.max_ts = 5
    m.physics.engine.find_equilibrium = False
    m.physics.engine.contact_solver = contact_solver.RETURN_MAPPING#local_iterations#flux_from_previous_iteration#return_mapping
    m.setup_contact_friction(m.physics.engine.contact_solver)
    if m.params.linear_type == sim_params.cpu_gmres_fs_cpr:
        m.physics.engine.update_uu_jacobian()

    # remove
    #for contact in m.physics.engine.contacts:
    #    contact.set_state(contact_state.TRUE_STUCK)

    m.physics.engine.t = 0.0
    restart_dt = -1
    #for ith_step, dt in enumerate(t):
    while m.physics.engine.t < max_t:
        dt_cur = m.reservoir.check_slope(dt, m.physics.engine.t + dt)
        m.params.first_ts = dt_cur
        m.params.max_ts = dt_cur
        run_python(m, dt_cur, restart_dt)
        m.reservoir.write_to_vtk(output_directory, ith_step + 1, m.physics)
        ith_step += 1
    m.print_timers()

if __name__ == '__main__':
    pass
    #just_run()