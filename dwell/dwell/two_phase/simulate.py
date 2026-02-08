import numpy as np
import pandas as pd
import time
import pickle

from dwell.two_phase.assemble_primary_variables_list import assemble_primary_variables_list
from dwell.two_phase.save_results import save_in_df, add_to_df

def simulate(initial_conditions, pipe_model, dt, bc, solver):
    # Assemble primary variables
    vars0 = assemble_primary_variables_list(initial_conditions, bc, pipe_model)

    # Initialize an empty DataFrame and store initial conditions in that DataFrame
    primary_variables_df = save_in_df(initial_primary_variables=vars0)
    # Initialize another empty data frame to store the phase props
    phase_props_df = pd.DataFrame()

    # Store the original dt entered by the user
    original_dt = np.array(dt, copy=True)

    """ Simulation main for-loop """
    print("Running simulation...")
    # Before updating vars, make a copy of it to ensure the perturbation does not carry over to the
    # next iteration unintentionally. Of course, here, without copying also works correctly.
    vars = np.array(vars0, copy=True)
    num_ts = len(dt)
    total_iter_counter = 0
    run_timer = 0
    for ts_counter in range(num_ts):
        vars0 = np.array(vars, copy=True)

        """ Newton-Raphson non-linear solver for-loop """
        ts_run_timer = 0
        iter_counter = 0
        simulation_timer = sum(dt[:ts_counter + 1])   # Simulation time up to now
        max_num_iter = solver['max_NR_iterations']
        while True:
            while iter_counter < max_num_iter:
            # for iter_counter in range(solver['max_NR_iterations']):
                start_time = time.time()
                residuals, jac = construct_Jacobian_matrix(pipe_model, vars0, vars, dt[ts_counter], bc,
                                                           simulation_timer, iter_counter, total_iter_counter)
                stop_time = time.time()
                iter_runtime = stop_time - start_time
                ts_run_timer += iter_runtime

                res_norm = np.linalg.norm(residuals)
                print("ts_counter = %d, iter_counter = %d, res = %e, iter_runtime = %e" % (ts_counter, iter_counter,
                                                                                           res_norm, iter_runtime))
                if res_norm < solver['NR_tolerance']:
                    total_iter_counter += iter_counter + 1

                    """ Store the primary vars in a DataFrame in which the initial conditions were stored """
                    primary_variables_df = add_to_df(primary_variables=vars, column_name=f'Time step {ts_counter}',
                                                     primary_variables_df=primary_variables_df)

                    # Update properties at the current time step with the original vars unaffected by eps_p, eps_temp, and eps_z
                    phase_props_df = pipe_model.calc_mass_energy_residuals(vars0, vars, dt[ts_counter], bc, simulation_timer, iter_counter, total_iter_counter, flag=-1, phase_props_df=phase_props_df)
                    break
                # elif iter_counter == solver['max_NR_iterations']-1 and res_norm > solver['NR_tolerance']:
                #     raise Exception("Solution did not converge!")

                increments = np.linalg.solve(jac, -residuals)
                vars += increments  # vars are updated here by adding the increments
                assert np.all(vars >= 0), "Negative primary variable(s) is observed!"

                iter_counter += 1  # Update the iteration counter

            else:
                # Max iterations reached, halve the time step
                dt[ts_counter] /= 2
                print("*** Maximum number of iterations reached. Halving the time step size to. " + str(dt[ts_counter]) + " second(s) ***")

                simulation_timer = sum(dt[:ts_counter + 1])  # Simulation time up to now
                # dt[ts_counter+1] = dt[ts_counter]   # Consider the next time step size the same as the current reduced time step size
                max_num_iter += solver['max_NR_iterations']
                vars = np.array(vars0, copy=True)
                continue  # Restart the while loop with a smaller time step

            # Break the outer while loop if converged within the inner while loop
            break

        print('------------------------------------------------------------------------- ts_run_timer =', ts_run_timer,
              ', simulation_timer =', simulation_timer)
        run_timer += ts_run_timer

    print('Total runtime = %5.3f sec, Total number of iterations = %d' % (run_timer, total_iter_counter))

    # Save the results stored in the data frame df
    primary_variables_df.to_pickle('stored_primary_variables.pkl')
    phase_props_df.to_pickle('stored_phase_props.pkl')

    # Save dt, component names, and the instance of PipeGeometry to a pickle file
    with open('stored_dt_pipe_geometry_other_info.pkl', 'wb') as file:
        pickle.dump((dt, pipe_model.pipe_geometry, pipe_model.fluid_model.components_names), file)

    if sum(original_dt) != sum(dt):
        print("** Note that due to time step cutting, the total time simulated by the simulator is less than the total simulation time \n"
              "entered by the user! Refer to dt stored in stored_dt_pipe_geometry_other_info.pkl to check the updated dt.")


def construct_Jacobian_matrix(pipe_model, vars0, vars, dt, bc, simulation_timer, iter_counter, total_iter_counter):
    num_segments = pipe_model.pipe_geometry.num_segments
    num_eqs = len(vars0)

    jac = np.zeros((num_eqs, num_eqs))
    residuals = pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                      total_iter_counter, flag=1)

    # Construct the Jacobian matrix
    for i in range(num_segments):
        # Derivatives of all the equations with respect to the pressure of segment i
        vars[i] += pipe_model.eps_p
        jac[:, i] = (pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                     total_iter_counter, flag=0) - residuals) / pipe_model.eps_p
        vars[i] -= pipe_model.eps_p

        for j in range(pipe_model.fluid_model.num_components - 1):
            # Derivatives of all the equations with respect to the mole fraction of component j in segment i
            vars[(j + 1) * num_segments + i] += pipe_model.eps_z
            jac[:, (j + 1) * num_segments + i] = (pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc,
                                                                                        simulation_timer,
                                                                                        iter_counter,
                                                                                        total_iter_counter,
                                                                                        flag=0) - residuals) / pipe_model.eps_z
            vars[(j + 1) * num_segments + i] -= pipe_model.eps_z

        if not pipe_model.isothermal:
            # Derivatives of all the equations with respect to the temperature of segment i
            vars[num_segments * pipe_model.fluid_model.num_components + i] += pipe_model.eps_temp
            jac[:, num_segments * pipe_model.fluid_model.num_components + i] = (pipe_model.calc_mass_energy_residuals(vars0, vars, dt, bc, simulation_timer, iter_counter,
                                                                        total_iter_counter, flag=0) - residuals) / pipe_model.eps_temp
            vars[num_segments * pipe_model.fluid_model.num_components + i] -= pipe_model.eps_temp

    # These three lines can be used to clean the Jacobian matrix (by setting the redundant values to 0), and
    # it does not affect the solution.
    # for i in range(num_segments):
    #     jac[i, i + 2:num_segments] = 0
    #     jac[i, num_segments + i + 2:] = 0

    return residuals, jac
