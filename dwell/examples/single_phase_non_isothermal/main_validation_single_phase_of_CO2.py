#%% Scenario description
# Pure CO2 injection
# Constant mass rate and temperature at the wellhead
# Wellbore is perforated at the lowest segment to the reservoir

#%% Import necessary modules and packages
from dwell.single_phase_non_isothermal.define_pipe_geometry import PipeGeometry
import dwell.single_phase_non_isothermal.define_fluid_model as define_fluid_model
from dwell.single_phase_non_isothermal.pipe_model import PipeModel
from dwell.single_phase_non_isothermal.set_initial_conditions import LinearAmbientTemperature, SingleAmbientTemperature
from dwell.single_phase_non_isothermal.add_bc_sources_sinks import WellLateralHeatTransfer, ConstantMassRateSource, Perforation
from dwell.single_phase_non_isothermal.add_bc_nodes import ConstantTempNode, ConstantPressureNode
from dwell.single_phase_non_isothermal.generate_rampup_time_steps import generate_rampup_time_steps
from dwell.single_phase_non_isothermal.simulate import simulate

from dwell.utilities.units import *
import dwell.utilities.library as library

import numpy as np

#%% Define pipe geometry
pipe_name = "Well1"
segments_lengths = 50 * meter() * np.ones(20)   # For a 1-kilometer well
pipe_ID = 0.1 * meter()
inclination_angle = 0   # in degrees relative to the vertical direction
wall_roughness = 2.5e-5 * meter()
verbose = True
pipe_geom = PipeGeometry(pipe_name, segments_lengths, pipe_ID, inclination_angle, wall_roughness, verbose)

#%% Define fluid model
pipe_name = "Well1"
components_names = ["CO2"]
# Flash-calculations
flash_calcs_obj = None   # Single-phase fluid flow
# Density
density_obj = define_fluid_model.Density_EoS_PT(components_names, EoS_name="PR")
# Enthalpy
# Method 1: Use Peng-Robinson EOS
enthalpy_obj = define_fluid_model.Enthalpy_PR_PT(components_names)
# Method 2: Use specific heat capacity at constant pressure
# Tref_for_cp = 175 * Kelvin()
# cp = 709 * Joule() / (kg() * Kelvin())   # CO2 cp at T = 300 K based on NIST is 37.22 J/mol*K or 845.72 J/kg*K ---- at T = 298 K based on NIST is 37.12 J/mol*K or 843.44 J/kg*K ---- at T = 175 K based on Engineering Toolbox is 709 J/kg*K
# enthalpy_obj = define_fluid_model.Enthalpy_cp(Tref_for_cp, cp)   # Enthalpy using specific heat capacity
# Internal energy
internal_energy_obj = define_fluid_model.InternalEnergy()
# Viscosity
viscosity_obj = define_fluid_model.Viscosity_Fenghouretal1998()
# IFT
IFT_obj = None   # Single-phase fluid flow

fluid_model = define_fluid_model.FluidModel(pipe_name, components_names, flash_calcs_obj, density_obj, enthalpy_obj,
                                            internal_energy_obj, viscosity_obj, IFT_obj, verbose)

#%% Create pipe model
pipe_model = PipeModel(pipe_geom, fluid_model, verbose=verbose, Cmax=1.2, Fv=1, eps_p=10 * Pascal(), eps_temp=0.1)

#%% Set initial conditions in the wellbore using LinearAmbientTemperature
pipe_name = "Well1"
WHP = 5 * bar()
fluid_comp = [1.]
WHT = 25 + 273.15 * Kelvin()
geothermal_gradient = 25 * Kelvin() / (kilo() * meter())

well_initial_conditions = LinearAmbientTemperature(pipe_name, pipe_geom, fluid_model, WHP, fluid_comp, WHT, geothermal_gradient, verbose)

#%% Set initial conditions in the pipe using SingleAmbientTemperature
# pipe_name = "Well1"
# ambient_temperature = 25 + 273.15 * Kelvin()
# pipe_head_pressure = 5 * bar()
# pipe_head_segment_index = pipe_geom.num_segments - 1   # index starts from zero
# fluid_comp = [1.]
# well_initial_conditions = SingleAmbientTemperature(pipe_name, pipe_geom, fluid_model, ambient_temperature,
#                                                    pipe_head_pressure, pipe_head_segment_index, fluid_comp, verbose)

#%% Put initial conditions in initial_conditions
initial_conditions = {'initial_pressure': well_initial_conditions.p_init_segments,
                      'initial_temperature': well_initial_conditions.temp_init_segments}

#%%  Add boundary conditions --> sources/sinks --> ConstantMassRateSource and Perforation
sources_sinks = {}

# Wellhead BC
pipe_name = "Well1"
segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
injection_mass_rate = 2 * kg() / second()   # Constant injection mass rate   # For a 1-kilometer well
injection_specific_enthalpy = 0 * Joule() / kg()  # Constant injection specific enthalpy   # For a 100-meter well
sources_sinks["ConstantMassRateSource"] = ConstantMassRateSource(pipe_name, pipe_geom, segment_index, start_time,
                                                                 stop_time, "inflow", injection_mass_rate,
                                                                 injection_specific_enthalpy, verbose)

# Bottomhole BC
pipe_name = "Well1"
segment_index = 0   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
reservoir_pressure = well_initial_conditions.p_init_segments[0] - 1
reservoir_cell_perm_list = [115 * milli() * Darcy(), 115 * milli() * Darcy()]   # [Kx, Ky]
# Note that the third element of reservoir_cell_size_list, which is the thickness of the reservoir cell
# (net pay thickness), must be equal to or smaller than the length of the wellbore segment that is connected
# to that reservoir cell.
reservoir_cell_size_list = [segments_lengths[segment_index], segments_lengths[segment_index], segments_lengths[segment_index]]   # [dx, dy, dz]
skin = 0
perforation_hole_diameter = 2 * centi() * meter()
sources_sinks["Perforation"] = Perforation(pipe_name, segment_index, start_time, stop_time, pipe_geom,
                                           reservoir_pressure, reservoir_cell_perm_list, reservoir_cell_size_list,
                                           perforation_hole_diameter, skin, verbose)

#%% Add boundary conditions --> sources/sinks --> WellLateralHeatTransfer
pipe_name = "Well1"
# Import rock data from the library
c_rock = library.mats_thermal_props['Rock']['c']
K_rock = library.mats_thermal_props['Rock']['K']
rho_rock = library.mats_thermal_props['Rock']['rho']
earth_thermal_props = {'T': well_initial_conditions.temp_init_segments, 'c': c_rock, 'K': K_rock, 'rho': rho_rock}
pipe_wall_thickness = 0.5 * centi() * meter()   # Thickness of the outermost layer of the wellbore
outermost_layer_OD = pipe_geom.pipe_ID + 2 * pipe_wall_thickness
# Set a constant overall heat transfer coefficient (Ui)
Ui = 0.2 * BTU() / (ft() ** 2 * hour() * Fahrenheit())  # Unit: BTU / (ft2 * hr * F)]  or  W / (m2 * C)
# sources_sinks["LateralHeatTransfer"] = WellLateralHeatTransfer(pipe_name, pipe_geom, earth_thermal_props,
#                                                                outermost_layer_OD, Ui, time_function_name="Chiu&Thakur",
#                                                                verbose=verbose)

#%% Add boundary conditions --> nodes (replace existing residual eqs.) --> ConstantTempNode and ConstantPressureNode
# Add constant temperature node at the wellhead
nodes = {}

pipe_name = "Well1"
segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
temperature = 20 + 273.15
nodes['ConstantTempNode'] = ConstantTempNode(pipe_name, segment_index, start_time, stop_time, temperature, verbose)

# Add constant pressure node at the wellhead
# pipe_name = "Well1"
# segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
# start_time = 0 * second()
# stop_time = "end"
# pressure = 10 * bar()
# nodes['ConstantPressureNode'] = ConstantPressureNode(pipe_name, segment_index, start_time, stop_time, pressure, verbose)

#%% Add boundary conditions --> extra nodes (extra residual equations)
extra_nodes = {}

#%% Store boundary conditions in bc
bc = {'source/sink': sources_sinks,
      'nodes': nodes,
      'extra_nodes': extra_nodes}

#%% Time stepping
dt = generate_rampup_time_steps(20, 1, 10 * minute())   # For a 1-kilometer well

#%% Solver: Newton-Raphson non-linear solver
max_num_NR_iterations = 50
NR_tolerance = 1e-5
solver = {'max_NR_iterations': max_num_NR_iterations, 'NR_tolerance': NR_tolerance}

#%% Start simulation
simulate(initial_conditions, pipe_model, dt, bc, solver)
