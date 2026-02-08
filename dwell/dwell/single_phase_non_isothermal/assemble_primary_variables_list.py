import numpy as np

def assemble_primary_variables_list(initial_conditions, bc):
    initial_pressure = initial_conditions['initial_pressure']
    initial_temperature = initial_conditions['initial_temperature']

    vars0 = np.concatenate((initial_pressure, initial_temperature))

    # if bc['extra_nodes'] is not None:
    #     # Add extra nodes to the list of the primary variables
    #     for node_name, node_object in bc['extra_nodes'].items():
    #         if node_name == 'PressureExtraNode':
    #             pass
    #         elif node_name == 'TempExtraNode':
    #             vars0 = np.append(vars0, node_object.temperature)
    #         elif node_name == 'ConcentrationExtraNode':
    #             pass

    return vars0
