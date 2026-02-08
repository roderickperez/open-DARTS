import numpy as np
import time
import pickle

from dwell.single_phase_non_isothermal.assemble_primary_variables_list import assemble_primary_variables_list
from dwell.single_phase_non_isothermal.save_results import save_in_df, add_to_df

def simulate(initial_conditions, pipe_model, dt, bc, solver):
    # Assemble primary variables
    vars0 = assemble_primary_variables_list(initial_conditions, bc)

    # Save dt and the instance of PipeGeometry to a pickle file
    with open('stored_dt_pipe_geometry.pkl', 'wb') as file:
        pickle.dump((dt, pipe_model.pipe_geometry), file)
    # Initialize an empty DataFrame and store initial conditions in a DataFrame
    primary_variables_df = save_in_df(initial_primary_variables=vars0)

    """ Simulation main for-loop """
    print("Running simulation...")
    # Before updating vars, make a copy of it to ensure the perturbation does not carry over to the
    # next iteration unintentionally. Of course, here, without copying also works correctly.
    vars = np.array(vars0, copy=True)
    num_ts = len(dt)
    total_iter_counter = 0
    run_timer = 0
    for ts_counter in range(num_ts):
        vars0 = np.array(vars)

        """ Newton-Raphson non-linear solver for-loop """
        ts_run_timer = 0
        simulation_timer = sum(dt[:ts_counter + 1])   # Simulation time up to now
        for iter_counter in range(solver['max_NR_iterations']):
            start_time = time.time()
            residuals, jac = construct_Jacobian_matrix(pipe_model, vars0, vars, dt[ts_counter], bc, simulation_timer,
                                                       iter_counter, total_iter_counter)
            stop_time = time.time()
            iter_runtime = stop_time - start_time
            ts_run_timer += iter_runtime

            res_norm = np.linalg.norm(residuals)
            print("ts_counter = %d, iter_counter = %d, res = %e, iter_runtime = %e" % (ts_counter, iter_counter,
                                                                                       res_norm, iter_runtime))
            if res_norm < solver['NR_tolerance']:
                total_iter_counter += iter_counter + 1

                """ Store the results in an Excel file or DataFrame in which the initial conditions were stored """
                primary_variables_df = add_to_df(primary_variables=vars, column_name=f'Time step {ts_counter}', primary_variables_df=primary_variables_df)

                # Update properties at the current time step with the original vars unaffected by eps_p and eps_temp
                pipe_model.calc_mass_energy_residuals(vars0, vars, dt[ts_counter], bc, simulation_timer, iter_counter,
                                                      total_iter_counter, flag=0)
                break
            elif iter_counter == solver['max_NR_iterations']-1 and res_norm > solver['NR_tolerance']:
                raise Exception("Solution did not converge!")

            increments = np.linalg.solve(jac, -residuals)
            vars += increments  # vars are updated here by adding the increments

        print('------------------------------------------------------------------------- ts_run_timer =', ts_run_timer,
              ', simulation_timer =', simulation_timer)
        run_timer += ts_run_timer

    print('Total runtime = %5.3f sec, Total number of iterations = %d' % (run_timer, total_iter_counter))

    # Save the results stored in the data frame df
    primary_variables_df.to_pickle('stored_primary_variables.pkl')


def construct_Jacobian_matrix(pipe_model, vars0, vars, dt, bc, simulation_timer, iter_counter, total_iter_counter):
    num_segments = pipe_model.pipe_geometry.num_segments
    num_eqs = int(len(vars0) / num_segments)

    jac = np.zeros((num_eqs * num_segments, num_eqs * num_segments))
    residuals = pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                      total_iter_counter, flag=1)

    for i in range(num_segments):
        # Derivatives of all the equations with respect to the pressure of segment i
        vars[i] += pipe_model.eps_p
        jac[:, i] = (pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                     total_iter_counter, flag=0) - residuals) / pipe_model.eps_p
        vars[i] -= pipe_model.eps_p

        # Derivatives of all the equations with respect to the temperature of segment i
        vars[num_segments + i] += pipe_model.eps_temp
        jac[:, num_segments + i] = (pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                                    total_iter_counter, flag=0) - residuals) / pipe_model.eps_temp
        vars[num_segments + i] -= pipe_model.eps_temp

    return residuals, jac
