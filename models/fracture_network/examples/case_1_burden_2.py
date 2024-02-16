import os
from .case_1_burden import input_data_case_1_burden

def input_data_case_1_burden_2():
    input_data = input_data_case_1_burden()
    input_data['case_name'] = 'case_1_burden_2'

    # second overburden layers (without fractures)
    input_data['overburden_2_thickness'] = input_data['height_res'] * 5
    input_data['overburden_2_layers'] = 1
    input_data['underburden_2_thickness'] = input_data['height_res'] * 5
    input_data['underburden_2_layers'] = 1

    return input_data