from darts.engines import value_vector
from model import Model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from darts.tools.hdf5_tools import load_hdf5_to_dict

def model_instance(output_folder = None):
    # intialize model
    m = Model()
    m.init(output_folder=output_folder)
    return m

def run_model_save_results():
    # intialize model
    m = model_instance()
    n_timesteps = 10

    # run and save well snapshots
    t = 0
    for i in range(n_timesteps):
        dt = 365
        m.run(dt)
        # m.save_data(type='solution', t=t+dt)
        t += dt
    m.print_timers()
    m.print_stat()

    # save legacy well data in *.pkl
    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    filename = os.path.join(m.output_folder, m.well_filename).split('.')[0] + '.pkl'
    td.to_pickle(filename)

    # save legacy well data in *.xlsx
    # writer = pd.ExcelWriter('time_data.xlsx')
    # td.to_excel(writer, 'Sheet1')
    # writer.close()

def find_one_array_in_another_indices(to_find, in_array):
    indices = []
    for element in to_find:
        id = np.where(in_array == element)[0]
        if id.size > 0:
            indices.append(id[0])
    return np.array(indices, dtype=np.intp)

def calc_connections_fluxes(data, evals, op_num, conn_ids, eval_ids = None, multipliers = None):
    # evaluate position of block_m, block_p in stored data, for every connection
    block_m = data['static']['block_m']
    block_p = data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[conn_ids], data['dynamic']['cell_id']) # well cells
    cell_p = find_one_array_in_another_indices(block_p[conn_ids], data['dynamic']['cell_id']) # reservoir cells
    assert(cell_m.size == len(conn_ids) and cell_p.size == len(conn_ids))

    nt = data['dynamic']['time'].size
    n_ops = int(evals[0].__class__.__name__.split('_')[-1])
    n_dim = int(evals[0].__class__.__name__.split('_')[-2])

    # support calculation with specific operators rather than with full
    if eval_ids is None:
        eval_ids = np.arange(n_ops)

    # support WI/CCF multipliers
    if multipliers is None:
        multipliers = np.ones((len(conn_ids), len(eval_ids)))

    # allocate data
    rates = np.zeros((nt, len(conn_ids), len(eval_ids)))
    id_state_cell = np.zeros(len(conn_ids), dtype=np.intp)

    state = value_vector()
    state.resize(n_dim)
    state_np = np.array(state, copy=False)
    values = value_vector()
    values.resize(n_ops)
    values_np = np.array(values, copy=False)

    # calculate fluxes at given time steps
    id_pres = data['dynamic']['variable_names'].index('pressure')
    for i in range(nt):

        pres = data['dynamic']['X'][i,:,id_pres]
        # estimate upwind cell indices for all connections
        p_diff = pres[cell_p] - pres[cell_m]
        down = p_diff < 0
        up = p_diff >= 0
        id_state_cell[down] = cell_m[down]
        id_state_cell[up] = cell_p[up]

        # looping over connections
        for j in range(len(conn_ids)):
            state_np[:] = data['dynamic']['X'][i, id_state_cell[j]]#[d['data']['pressure'][id_state_cell[j]], d['data']['temperature'][id_state_cell[j]]]
            evals[op_num[id_state_cell[j]]].evaluate(state, values)
            rates[i, j] = values_np[eval_ids] * multipliers[j] * p_diff[j]

    return data['dynamic']['time'], rates

def compare_well_rates():
    # load new well data
    m = model_instance()
    new_filename = os.path.join(m.output_folder, m.well_filename)
    new_data = load_hdf5_to_dict(new_filename)

    # calculate fluxes at all perforations
    # eval = mass_evaluator(m.physics.property_containers[0])
    # eval = engine_evaluator(m.physics.property_containers[0])
    evals = m.op_list
    eval_ids = np.arange(m.physics.n_vars, m.physics.n_vars + m.physics.n_vars * m.physics.nph)
    perfs = [p for well in m.reservoir.wells for p in well.perforations]
    conn_id_perforations = m.find_conn_id_for_perforation(perfs=perfs)
    perfs_ccf = np.array([p[2] for well in m.reservoir.wells for p in well.perforations])
    perfs_ccf_d = np.array([p[3] for well in m.reservoir.wells for p in well.perforations])
    multipliers = np.repeat(perfs_ccf[:, np.newaxis], eval_ids.size, axis=1)
    new_time, new_data = calc_connections_fluxes(data=new_data, evals=evals, op_num=m.op_num,
                                                 conn_ids=conn_id_perforations, eval_ids=eval_ids, multipliers=multipliers)

    # load old well data
    old_filename = os.path.join(m.output_folder, m.well_filename).split('.')[0] + '.pkl'
    old_data = pd.read_pickle(old_filename)
    old_time = old_data['time'].to_numpy()

    # old data
    counter = 0
    old_data_start_id = 0
    for well in m.reservoir.wells:
        for perf in well.perforations:
            fig, rate = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            pattern1 = well.name + ' : p ' + str(perf[0]) + ' c 0 rate (Kmol/day)'
            pattern2 = well.name + ' : p ' + str(perf[0]) + ' c 1 rate (Kmol/day)'
            nc = 2
            id_c0 = np.arange(0, new_data.shape[-1], nc)
            old = old_data[pattern1].to_numpy()
            new = -np.sum(new_data[:, counter, id_c0], axis=1)
            rate.plot(old_time[old_data_start_id:], old[old_data_start_id:],
                      color='b', marker='o', markersize=5, label='Legacy code')
            rate.plot(new_time, new, color='r', marker='*', markersize=5, label='New python code')
            rate.legend(loc='upper right', prop={'size': 18})
            rate.set_xlabel(r'time [day]', fontsize=16)
            rate.set_ylabel(r'water + steam rate [Kmol/day]', fontsize=16)
            fig.tight_layout()
            plt.savefig(well.name + '_p_' + str(perf[0]) + '_rate_cmp.png')
            counter += 1

if __name__ == '__main__':
    pass
    #run_model_save_results()
    #compare_well_rates()