"""
Scenario description:
    Non-isothermal miscible 2-phase fluid flow with variable K values
    Impure CO2 injection into a wellbore partially full of water
    Constant mass rate at the wellhead
    Wellbore is perforated at the lowest segment to the reservoir

Notes:
    I reduced the Newton tolerance from 1e-5 to 1e-3 to make the solver converge, otherwise, it converges after a lot of
    iterations.
    This scenario needs ramp-up time stepping (needs small time steps at the beginning due to intense transients at the beginning)
"""

#%% Import necessary modules and packages
from dwell.two_phase.define_pipe_geometry import PipeGeometry
import dwell.two_phase.define_fluid_model as define_fluid_model
from dwell.two_phase.pipe_model import PipeModel
from dwell.two_phase.set_initial_conditions import LinearAmbientTemperature
from dwell.two_phase.check_initial_conditions import check_initial_conditions
from dwell.two_phase.add_bc_sources_sinks import ConstantMassRateSource, Perforation, WellLateralHeatTransfer
from dwell.two_phase.add_bc_nodes import ConstantTempNode, ConstantPressureNode
from dwell.two_phase.add_bc_extra_nodes import ConstantPressureExtraNode
from dwell.two_phase.generate_rampup_time_steps import generate_rampup_time_steps
from dwell.two_phase.simulate import simulate

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData

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

#%% Define fluid model, and then property evaluator
pipe_name = "Well1"
# The primary variables of composition will be stored in the order of the names of the components below. Of course,
# the mole fraction of one of the components is not considered as a primary variable.
components_names = ['CO2', 'C1', 'H2O']
phases_names = ['gas', 'liquid']   # gas is the CO2-CH4-rich phase and liquid is the water-rich (aqueous) phase

# Flash calculations
# Method 1: Constant K values
# flash_calcs_eval = define_fluid_model.ConstantK([4, 1e-1])

# Method 2: K values as a function of pressure, temp, and composition
comp_data = CompData(components_names, setprops=True)

pr = CubicEoS(comp_data, CubicEoS.PR)
# aq = Jager2003(comp_data)
aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

flash_params = FlashParams(comp_data)

# EoS-related parameters
flash_params.add_eos("PR", pr)
flash_params.add_eos("AQ", aq)

flash_calcs_eval = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])

# Density
# density_eval = dict([('gas', define_fluid_model.DensityBasic(compr=1e-8/Pascal(), rho_ref=200*kg()/meter()**3, p_ref=1e5*Pascal())),
#                      ('liquid', define_fluid_model.DensityBasic(compr=1e-10/Pascal(), rho_ref=1000*kg()/meter()**3, p_ref=1e5*Pascal()))])
density_eval = dict([('gas', define_fluid_model.Density_EoS_PT(components_names, EoS_name="PR")),
                     ('liquid', define_fluid_model.Density_Garcia2001(components_names))])

# Enthalpy
# Method 1: Use Peng-Robinson EOS
enthalpy_eval = dict([('gas', define_fluid_model.Enthalpy_PR_PT(components_names)),
                      ('liquid', define_fluid_model.Enthalpy_AQ_PT(components_names))])
# Method 2: Use specific heat capacity at constant pressure
# Tref_for_cp = 175 * Kelvin()
# cp = 709 * Joule() / (kg() * Kelvin())   # CO2 cp at T = 300 K based on NIST is 37.22 J/mol*K or 845.72 J/kg*K ---- at T = 298 K based on NIST is 37.12 J/mol*K or 843.44 J/kg*K ---- at T = 175 K based on Engineering Toolbox is 709 J/kg*K
# enthalpy_eval = define_fluid_model.Enthalpy_cp(Tref_for_cp, cp)   # Enthalpy using specific heat capacity

# Internal energy
internal_energy_eval = define_fluid_model.InternalEnergy()

# Viscosity
# viscosity_eval = dict([('gas', define_fluid_model.ConstFunc(0.05*centi()*Poise())),
#                        ('liquid', define_fluid_model.ConstFunc(0.5*centi()*Poise()))])
viscosity_eval = dict([('gas', define_fluid_model.Viscosity_Fenghouretal1998()),
                       ('liquid', define_fluid_model.Viscosity_Islam2012(components_names))])

# IFT
# IFT_eval = define_fluid_model.IFT_MCM(components_names[0])
IFT_eval = define_fluid_model.IFT_multicomponent_MCM(components_names)
# IFT_eval = define_fluid_model.ConstFunc(0.05 * Newton() / meter())

# Relative permeability
rel_perm_eval = dict([('gas', define_fluid_model.PhaseRelPerm_BC("gas", 0, 0)),
                      ('liquid', define_fluid_model.PhaseRelPerm_BC("liquid", 0, 0))])

fluid_model = define_fluid_model.FluidModel(pipe_name, components_names, phases_names, flash_calcs_eval, density_eval,
                                            viscosity_eval, IFT_eval, rel_perm_eval, enthalpy_eval, internal_energy_eval, verbose=verbose)

property_evaluator = define_fluid_model.PropertyEvaluator(fluid_model)

#%% Create pipe model
isothermal = False
pipe_model = PipeModel(pipe_geom, fluid_model, property_evaluator, isothermal=isothermal,
                       Cmax=1.2, Fv=1, eps_p=10 * Pascal(), eps_temp=0.1, eps_z=0.00001, verbose=verbose)

#%% Set initial conditions in the wellbore using LinearAmbientTemperature
pipe_name = "Well1"
pipe_head_pressure = 5 * bar()  # Equivalent to wellhead pressure
pipe_head_temperature = 25 + 273.15 * Kelvin()   # Equivalent to wellhead temperature
temp_grad = 25 * Kelvin() / (kilo() * meter())   # Equivalent to geothermal gradient
pipe_head_segment_index = pipe_geom.num_segments - 1   # index starts from zero

initial_water_comp_bottom = [1e-5, 1e-5, 1 - 2 * 1e-5]
initial_gas_comp_top = [0.01, 0.98, 0.01]

initial_fluid_conditions = {'phases_names': ['liquid', 'gas'], 'phases_compositions': [initial_water_comp_bottom, initial_gas_comp_top],
                            'pipe_intervals': [[0, 300], [300, 1000]]}   # 0 is the beginning of the pipe and pipe_intervals are TVD

well_initial_conditions = LinearAmbientTemperature(pipe_name, pipe_geom, fluid_model, pipe_head_pressure,
                                                   pipe_head_temperature, temp_grad, pipe_head_segment_index,
                                                   initial_fluid_conditions, verbose)

#%% Put initial conditions in initial_conditions
initial_CO2_mole_fraction = np.concat(([initial_water_comp_bottom[0]] * 6, [initial_gas_comp_top[0]] * 14))
initial_C1_mole_fraction = np.concat(([initial_water_comp_bottom[1]] * 6, [initial_gas_comp_top[1]] * 14))

initial_conditions = {'initial_pressure': well_initial_conditions.p_init_segments,
                      'initial_CO2_mole_fraction': initial_CO2_mole_fraction,
                      'initial_C1_mole_fraction': initial_C1_mole_fraction,
                      'initial_temperature': well_initial_conditions.temp_init_segments}
check_initial_conditions(initial_conditions, components_names, isothermal)

#%%  Add boundary conditions --> sources/sinks --> ConstantMassRateSource and Perforation
sources_sinks = {}

# Wellhead BC
pipe_name = "Well1"
segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
injection_mass_rate = 10 * kg() / second()

flow_direction = ["inflow", "along the pipe"]

injected_fluid_mole_fractions = [0.95, 0.04, 0.01]   # Impure CO2 injection

injected_fluid_specific_enthalpy = -10000 * Joule() / kg()  # Constant injection specific enthalpy
sources_sinks["ConstantMassRateSource"] = ConstantMassRateSource(pipe_name, pipe_geom, fluid_model, segment_index, start_time,
                                                                 stop_time, flow_direction, injection_mass_rate, isothermal,
                                                                 injected_fluid_mole_fractions,
                                                                 injected_fluid_specific_enthalpy, verbose=verbose)

# Bottomhole BC
pipe_name = "Well1"
segment_index = 0   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
reservoir_pressure = well_initial_conditions.p_init_segments[segment_index] - 1
reservoir_fluid_mole_fractions = [1e-5, 1e-5, 1 - 2 * 1e-5]
reservoir_cell_perm_list = [100 * milli() * Darcy(), 100 * milli() * Darcy()]   # [Kx, Ky]
# Note that the third element of reservoir_cell_size_list, which is the thickness of the reservoir cell
# (net pay thickness), must be equal to or smaller than the length of the wellbore segment that is connected
# to that reservoir cell.
reservoir_cell_size_list = [segments_lengths[segment_index], segments_lengths[segment_index], segments_lengths[segment_index]]   # [dx, dy, dz]
skin = 0
perforation_hole_diameter = 2 * centi() * meter()   # Perforation hole diameter is needed for the kinetic energy and momentum calculations
sources_sinks["Perforation"] = Perforation(pipe_name, segment_index, start_time, stop_time, pipe_geom, fluid_model,
                                           reservoir_pressure, reservoir_fluid_mole_fractions, reservoir_cell_perm_list,
                                           reservoir_cell_size_list, perforation_hole_diameter, skin=skin, verbose=verbose)

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
sources_sinks["LateralHeatTransfer"] = WellLateralHeatTransfer(pipe_name, pipe_geom, earth_thermal_props,
                                                               outermost_layer_OD, Ui, time_function_name="Chiu&Thakur",
                                                               verbose=verbose)

#%% Add boundary conditions --> nodes (replace existing residual eqs.) --> ConstantTempNode and ConstantPressureNode
# Add constant temperature node at the wellhead
nodes = {}

# pipe_name = "Well1"
# segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
# start_time = 0 * second()
# stop_time = "end"
# temperature = 20 + 273.15
# nodes['ConstantTempNode'] = ConstantTempNode(pipe_name, segment_index, start_time, stop_time, temperature, verbose)

# Add constant pressure node at the wellhead
# pipe_name = "Well1"
# segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
# start_time = 0 * second()
# stop_time = "end"
# pressure = 10 * bar()
# nodes['ConstantPressureNode'] = ConstantPressureNode(pipe_name, segment_index, start_time, stop_time, pressure, verbose)

#%% Add boundary conditions --> extra nodes (extra residual equations)
extra_nodes = {}

# # Add constant pressure extra node at the wellhead
# pipe_name = "Well1"
# segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
# start_time = 0 * second()
# stop_time = "end"
# pressure = 1 * bar()
# flow_direction = "outflow"
# extra_nodes['ConstantPressureExtraNode'] = ConstantPressureExtraNode(pipe_name, segment_index, start_time, stop_time, pressure, flow_direction, initial_conditions, verbose)
# initial_conditions = extra_nodes['ConstantPressureExtraNode'].initial_conditions

#%% Store boundary conditions in bc
bc = {'source/sink': sources_sinks,
      'nodes': nodes,
      'extra_nodes': extra_nodes}

#%% Time stepping
dt = generate_rampup_time_steps(20, 1, 600)

#%% Solver: Newton-Raphson non-linear solver
max_num_NR_iterations = 50
NR_tolerance = 1e-3
solver = {'max_NR_iterations': max_num_NR_iterations, 'NR_tolerance': NR_tolerance}

#%% Start simulation
simulate(initial_conditions, pipe_model, dt, bc, solver)
