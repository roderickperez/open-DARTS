import numpy as np

def generate_rampup_time_steps(num_rampup_ts: int, target_ts: float, simulation_time: float):
    """
    This function generates a time-step sequence that ramps up geometrically until it reaches the given
    target time step. The rest of the interval is then divided into a number of target time steps.
    Note that the final time step may be shorter than the target time step in order to exactly reach
    the simulation time.

    :param num_rampup_ts: Number of time steps till the target time step
    :type num_rampup_ts: int
    :param target_ts: Target time step after initial ramp-up [second]
    :type target_ts: float
    :param simulation_time: Total simulation time [second]
    :type simulation_time: float

    :return: A list of time steps [second]
    :rtype: list of floats
    """
    # Initial geometric series
    dt_init = (target_ts / 2.0 ** np.concatenate(([num_rampup_ts], np.arange(num_rampup_ts, 0, -1))))
    cs_time = np.cumsum(dt_init)

    if np.any(cs_time > simulation_time):
        dt_init = dt_init[cs_time < simulation_time]

    # Remaining time that must be discretized
    dt_left = simulation_time - np.sum(dt_init)

    # Even steps
    dt_rem = np.full(int(np.floor(dt_left / target_ts)), target_ts)

    # Final ministep if present
    dt_final = simulation_time - np.sum(dt_init) - np.sum(dt_rem)

    # Less than to account for rounding errors leading to a very small negative time-step.
    if dt_final <= 0:
        dt_final = np.array([])
    else:
        dt_final = np.array([dt_final])

    # Combined timesteps
    dT = np.concatenate((dt_init, dt_rem, dt_final))

    return dT.tolist()
