import os
from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil
from model_CO2 import ModelCCS

from cases.case_base import set_input_data_base
from cases.case_geothermal import set_input_data_geothermal, set_input_data_well_controls_geothermal
from cases.case_deadoil import set_input_data_deadoil, set_input_data_well_controls_deadoil
from cases.case_co2 import set_input_data_co2, set_input_data_well_controls_co2
from cases.case_geom_generate import set_input_data_geom_generate
from cases.case_geom_grdecl import set_input_data_geom_grdecl

from output_functions import output_vtk, output_time_data

# make a choice between different reservoirs
match 2:
    case 1: case_geom = 'generate_5x3x4'
    case 2: case_geom = 'generate_51x51x1'
    case 3: case_geom = 'generate_51x51x1_faultmult'
    case 4: case_geom = 'generate_100x100x100'
    # for grdecl cases, the grid and properties files must be in meshes/<case> folder, the last '_' part is ignored so can use it for naming
    case 5: case_geom = '40x40x10'
    case 6: case_geom = 'brugge'
    case 7: case_geom = 'brugge_noburdenlayers'
# make a choice between different physics
match 1:
    case 1: m = ModelGeothermal()
    case 2: m = ModelDeadOil()
    case 3: m = ModelCCS()
# make a choice between different well controls
match 3:
    case 1: well_controls = 'rate'
    case 2: well_controls = 'bhp'
    case 3: well_controls = 'periodic'


case = case_geom + '_' + well_controls
out_dir = os.path.join('results', m.physics_type + '_' + case)

# 1. set physics-specific input data
match m.physics_type:
    case 'geothermal': m.idata = set_input_data_geothermal()
    case 'deadoil': m.idata = set_input_data_deadoil()
    case 'CCS': m.idata = set_input_data_co2()

# 2. set default input data, generic parameters
# including time stepping and convergence parameters, boundary conditions and rock properties for all cases and default rock properties
set_input_data_base(m.idata, case_geom)

# 3. set grid size and resolution, and add wells
if 'generate' in case_geom:
    set_input_data_geom_generate(m.idata, case)
else:
    set_input_data_geom_grdecl(m.idata, case)

# 4. set well controls
match m.physics_type:
    case 'geothermal': set_input_data_well_controls_geothermal(m.idata, case)
    case 'deadoil': set_input_data_well_controls_deadoil(m.idata, case)
    case 'CCS': set_input_data_well_controls_co2(m.idata, case)

if 'geothermal' not in m.physics_type:
    m.idata.geom.burden_layers = 0

if 'CCS' not in m.physics_type:
    m.idata.sim.DataTS.dt_first = 1e-5

# now, the data is set to m.idata and will be taken there

os.makedirs(out_dir, exist_ok=True)
if 1:
    from darts.engines import redirect_darts_output
    log_filename = os.path.join(out_dir, 'run.log')
    log_stream = redirect_darts_output(log_filename)

print('----- Test started', 'physics_type:', m.physics_type, 'case:', case, ' ------')

m.set_physics()

arrays = m.init_input_arrays()
# custom arrays can be read here
# arrays['new_array_name'] = read_float_array(filename, 'new_array_name')
# arrays['new_array_name'] = read_int_array(filename, 'new_array_name')
# also rock properties such as permeability can be modified here if needed
# arrays['PERMX'] *= 0.1
m.init_reservoir(arrays=arrays)

# time stepping and convergence parameters
m.set_sim_params_data_ts(data_ts=m.idata.sim.DataTS)

m.timer.node["initialization"].stop()

m.init()
m.set_output(output_folder=out_dir, all_phase_props=True, verbose = True)
m.set_well_controls_idata()

m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_init'))

ret = m.run_simulation()

m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_last'))
m.print_timers()

print("Writing vtk files...")
output_vtk(out_dir, m)
print("Finished vtk files writing")

output_time_data(out_dir, m, case)

print("Computation of the case", case, "completed")


