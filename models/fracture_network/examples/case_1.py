import os
from .input_default import input_data_default

def input_data_case_1():
    input_data = input_data_default()

    input_data['case_name'] = 'case_1'

    # geometry
    input_data['frac_file'] = os.path.join('examples', 'frac_1.txt')

    # do not use cleaned mesh
    input_data['mesh_prefix'] = 'raw_lc'

    # cell sizes
    input_data['char_len'] = 50  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = 150  # grid size near grid boundaries [m]
    input_data['char_len_well'] = 100  # grid size near wells [m]

    # initial pressure and temperature
    # uniform initial pressure and temperature
    input_data['initial_uniform'] = True
    input_data['uniform_pressure'] = 350.  # bar
    input_data['uniform_temperature'] = 348.15  # K

    # well locations
    input_data['inj_well_coords'] = [[100, 200, 25]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[800, 800, 25]]

    # well in the matrix cells or in the fractures
    input_data['well_loc_type'] = 'wells_in_frac'

    # extrusion - number of layers by Z axis
    input_data['rsv_layers'] = 3

    input_data['z_top'] = 2000  # [m]
    input_data['height_res'] = 50  # [m]

    return input_data


