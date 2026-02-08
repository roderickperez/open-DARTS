import numpy as np


def assemble_primary_variables_list(initial_conditions, bc, pipe_model):
    vars0 = initial_conditions['initial_pressure']
    # With this method, the order of storage of initial conditions in initial_conditions is not important, and the
    # order of the primary variables is always based on the order of the list of the names of the components.
    for component_name in pipe_model.fluid_model.components_names:
        try:
            initial_comp_mole_fraction = initial_conditions['initial_' + component_name + '_mole_fraction']
            vars0 = np.concatenate((vars0, initial_comp_mole_fraction))
        except:
            pass

    if not pipe_model.isothermal:
        vars0 = np.concatenate((vars0, initial_conditions['initial_temperature']))

    if bc['extra_nodes']:   # If bc['extra_nodes'] is not empty, this condition is satisfied.
        # Add extra nodes to the list of the primary variables
        for extra_node_name, extra_node_object in bc['extra_nodes'].items():
            if extra_node_name == 'ConstantPressureExtraNode':
                vars0 = np.concatenate((vars0, [extra_node_object.ghost_segment_pressure]))
            # elif extra_node_name == 'ConstantTempExtraNode':
            #     vars0 = np.append(vars0, extra_node_object.ghost_segment_temperature)
            # elif extra_node_name == 'ConstantConcentrationExtraNode':
            #     vars0 = np.append(vars0, extra_node_object.ghost_segment_concentration)

    return vars0
