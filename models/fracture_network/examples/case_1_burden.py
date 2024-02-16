import os
from .case_1 import input_data_case_1

def input_data_case_1_burden():
    input_data = input_data_case_1()
    input_data['case_name'] = 'case_1_burden'

    # overburden layers (with fractures)
    input_data['overburden_thickness'] = input_data['height_res'] * 2
    input_data['overburden_layers'] = 1
    input_data['underburden_thickness'] = input_data['height_res'] * 2
    input_data['underburden_layers'] = 1

    return input_data
