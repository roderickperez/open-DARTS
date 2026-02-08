"""
Scenario description:
    Isothermal miscible 2-phase fluid flow with variable K values
    Impure CO₂ injection into a wellbore half-filled with water
    Constant mass rate at the wellhead
    Wellbore is perforated at the lowest segment to the reservoir

Notes:
    Solution ultimately diverges if densities are calculated using EOS. This divergence happens because when the pressure
    of the CO2-rich phase goes beyond the phase saturation pressure, the density of the CO2-rich phase changes
    instantaneously.
    This divergence can be prevented using the following methods:
    - The system temperature is set to a value higher than the CO2 critical temperature
    - Instead of CO2, methane is used.
    - The pressure of the system during the simulation must be kept below the saturation pressure.
"""

#%% Import necessary modules and packages
from dwell.two_phase.define_pipe_geometry import PipeGeometry
import dwell.two_phase.define_fluid_model as define_fluid_model
from dwell.two_phase.pipe_model import PipeModel
from dwell.two_phase.set_initial_conditions import SingleAmbientTemperature
from dwell.two_phase.check_initial_conditions import check_initial_conditions
from dwell.two_phase.add_bc_sources_sinks import ConstantMassRateSource, Perforation
from dwell.two_phase.add_bc_nodes import ConstantTempNode, ConstantPressureNode
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
components_names = ['CO2', 'H2O']   # For CO2 injection
# components_names = ['C1', 'H2O']   # For methane injection
phases_names = ['gas', 'liquid']   # gas is the CO2-rich phase and liquid is the water-rich (aqueous) phase

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
# enthalpy_obj = define_fluid_model.Enthalpy_PR_PT(components_names)
# Method 2: Use specific heat capacity at constant pressure
# Tref_for_cp = 175 * Kelvin()
# cp = 709 * Joule() / (kg() * Kelvin())   # CO2 cp at T = 300 K based on NIST is 37.22 J/mol*K or 845.72 J/kg*K ---- at T = 298 K based on NIST is 37.12 J/mol*K or 843.44 J/kg*K ---- at T = 175 K based on Engineering Toolbox is 709 J/kg*K
# enthalpy_obj = define_fluid_model.Enthalpy_cp(Tref_for_cp, cp)   # Enthalpy using specific heat capacity

# Internal energy
# internal_energy_obj = define_fluid_model.InternalEnergy()

# Viscosity
# viscosity_eval = dict([('gas', define_fluid_model.ConstFunc(0.05*centi()*Poise())),
#                        ('liquid', define_fluid_model.ConstFunc(0.5*centi()*Poise()))])
viscosity_eval = dict([('gas', define_fluid_model.Viscosity_Fenghouretal1998()),
                       ('liquid', define_fluid_model.Viscosity_Islam2012(components_names))])

# IFT
# IFT_obj = define_fluid_model.IFT_MCM(components_names[0])
IFT_eval = define_fluid_model.ConstFunc(0.05 * Newton() / meter())

# Relative permeability
rel_perm_eval = dict([('gas', define_fluid_model.PhaseRelPerm_BC("gas", 0, 0)),
                      ('liquid', define_fluid_model.PhaseRelPerm_BC("liquid", 0, 0))])

fluid_model = define_fluid_model.FluidModel(pipe_name, components_names, phases_names, flash_calcs_eval, density_eval,
                                            viscosity_eval, IFT_eval, rel_perm_eval, verbose=verbose)

property_evaluator = define_fluid_model.PropertyEvaluator(fluid_model)

#%% Create pipe model
isothermal = True
system_temperature = 40 + 273.15 * Kelvin()
pipe_model = PipeModel(pipe_geom, fluid_model, property_evaluator, isothermal=isothermal, system_temperature=system_temperature,
                       Cmax=1.2, Fv=1, eps_p=10 * Pascal(), eps_temp=0.1, eps_z=0.00001, verbose=verbose)

#%% Set initial conditions in the wellbore using WellInitialConditions
# pipe_name = "Well1"
# WHP = 5 * bar()
# fluid_comp = [1.]
# WHT = 25 + 273.15 * Kelvin()
# geothermal_gradient = 0 * Kelvin() / (kilo() * meter())
#
# well_initial_conditions = WellInitialConditions(pipe_name, pipe_geom, fluid_model, WHP, fluid_comp, WHT, geothermal_gradient, verbose)

#%% Set initial conditions in the pipe using SingleAmbientTemperature
pipe_name = "Well1"
ambient_temperature = system_temperature
pipe_head_pressure = 5 * bar()
pipe_head_segment_index = pipe_geom.num_segments - 1   # index starts from zero

initial_fluid_comp_bottom_half = [1e-10, 1 - 1e-10]
initial_fluid_comp_top_half = [0.99, 0.01]

initial_fluid_conditions = {'phases_names': ['liquid', 'gas'], 'phases_compositions': [initial_fluid_comp_bottom_half, initial_fluid_comp_top_half],
                            'pipe_intervals': [[0, 500], [500, 1000]]}   # 0 is the beginning of the pipe and pipe_intervals are TVD

well_initial_conditions = SingleAmbientTemperature(pipe_name, pipe_geom, fluid_model, ambient_temperature,
                                                   pipe_head_pressure, pipe_head_segment_index, initial_fluid_conditions, verbose)

#%% Put initial conditions in initial_conditions
initial_CO2_mole_fraction = np.concat(([initial_fluid_comp_bottom_half[0]] * int(pipe_geom.num_segments/2), [initial_fluid_comp_top_half[0]] * int(pipe_geom.num_segments/2)))

initial_conditions = {'initial_pressure': well_initial_conditions.p_init_segments,
                      'initial_CO2_mole_fraction': initial_CO2_mole_fraction}

# Methane
# initial_C1_mole_fraction = np.concat(([initial_fluid_comp_bottom_half[0]] * int(pipe_geom.num_segments/2), [initial_fluid_comp_top_half[0]] * int(pipe_geom.num_segments/2)))
#
# initial_conditions = {'initial_pressure': well_initial_conditions.p_init_segments,
#                       'initial_C1_mole_fraction': initial_C1_mole_fraction}

check_initial_conditions(initial_conditions, components_names, isothermal)

#%%  Add boundary conditions --> sources/sinks --> ConstantMassRateSource and Perforation
sources_sinks = {}

# Wellhead BC
pipe_name = "Well1"
segment_index = pipe_geom.num_segments - 1   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
injection_mass_rate = 25 * kg() / second()

flow_direction = ["inflow", "along the pipe"]

injected_fluid_mole_fractions = [0.99, 0.01]

# injection_specific_enthalpy = -20000 * Joule() / kg()  # Constant injection specific enthalpy
sources_sinks["ConstantMassRateSource"] = ConstantMassRateSource(pipe_name, pipe_geom, fluid_model, segment_index, start_time,
                                                                 stop_time, flow_direction, injection_mass_rate, isothermal,
                                                                 injected_fluid_mole_fractions, verbose=verbose)

# Bottomhole BC
pipe_name = "Well1"
segment_index = 0   # 0 is the index of the first segment
start_time = 0 * second()
stop_time = "end"
reservoir_pressure = well_initial_conditions.p_init_segments[segment_index] - 1
reservoir_fluid_mole_fractions = [1e-10, 1 - 1e-10]
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

#%% Store boundary conditions in bc
bc = {'source/sink': sources_sinks,
      'nodes': nodes,
      'extra_nodes': extra_nodes}

#%% Time stepping
dt = generate_rampup_time_steps(20, 1, 250)
# dt = 1 * second() * np.ones(250)   # For methane

#%% Solver: Newton-Raphson non-linear solver
max_num_NR_iterations = 50
NR_tolerance = 1e-5
solver = {'max_NR_iterations': max_num_NR_iterations, 'NR_tolerance': NR_tolerance}

#%% Start simulation
simulate(initial_conditions, pipe_model, dt, bc, solver)
