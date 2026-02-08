import numpy as np

def input_data_default():
    input_data = dict()

    input_data['inj_well_coords'] = [[500, 1000, 2000]]
    input_data['prod_well_coords'] = [[3500, 3000, 2000]] 
    input_data['x1']=0
    input_data['x2']=4000
    input_data['y1']=900
    input_data['y2']=3100 

    # # DFN framework parameters (for mesh generation)

    input_data['frac_file'] = 'frac.txt'  # fracture tips coordinates X1 Y1 X2 Z2; should contain at least 2 rows (2 fractures)
    #input_data['char_len']=200
    input_data['char_len']=25
    input_data['merge_threshold']=.86    # [0.5  --  0.86],   #  200*.5= 100
    input_data['box_data']=[
    [input_data['x1'], input_data['y1']],  # bottom left
    [input_data['x2'], input_data['y1']],  # bottom right
    [input_data['x2'], input_data['y2']],  # top right
    [input_data['x1'], input_data['y2']]   # top left
    ]


    # # extrusion - number of layers by Z axis
    input_data['rsv_layers'] = 1

    # # no overburden layers (fractured) by default
    input_data['overburden_thickness'] = 0
    input_data['overburden_layers'] = 0
    input_data['underburden_thickness'] = 0
    input_data['underburden_layers'] = 0

    # # no second overburden layers (without fractures) by default
    input_data['overburden_2_thickness'] = 0
    input_data['overburden_2_layers'] = 0
    input_data['underburden_2_thickness'] = 0
    input_data['underburden_2_layers'] = 0


    return input_data
