# from reservoir import StructReservoir
from phreeqc_dissolution.conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions, convert_rate, bar2atm
from phreeqc_dissolution.physics import PhreeqcDissolution

from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.super.property_container import PropertyContainer
from darts.engines import *

import numpy as np
import pickle
import os
from math import fabs
import warnings

try:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

# Definition of your input parameter data structure,
# change as you see fit (when you need more constant values, etc.)!!
class MyOwnDataStruct:
    def __init__(self, nc, zmin, temp, stoich_matrix, pressure_init, c_r, kin_fact,  exp_w=1, exp_g=1):
        """
        Data structure class which holds various input parameters for simulation
        :param nc: number of components used in simulation
        :param zmin: actual 0 used for composition (ussualy >0, around some small epsilon)
        :param temp: temperature
        """
        self.num_comp = nc
        self.min_z = zmin
        self.temperature = temp
        self.stoich_matrix = stoich_matrix
        self.exp_w = exp_w
        self.exp_g = exp_g
        self.pressure_init = pressure_init
        self.c_r = c_r
        self.kin_fact = kin_fact

# Actual Model class creation here!
class Model(DartsModel):
    def __init__(self, nx):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir(nx=nx)
        self.set_physics()

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-7
        self.params.max_ts = 1e-4

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 20
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        self.params.newton_params[0] = 0.2
        # self.params.norm_type = 1

        self.runtime = 1
        self.timer.node["initialization"].stop()

    def set_physics(self):
        # some properties
        self.temperature = 323.15           # K
        self.pressure_init = 100            # bar
        self.inj_rate = convert_rate(0.153742)     # input: ml/min; output: m3/day
        self.c_r = 1e-6
        # self.kin_fact = (1 + self.c_r * (self.pressure_init - 1)) * 2710 / 0.1000869 * np.mean(self.solid_sat)
        self.kin_fact = 1
        self.comp_min = 1e-11
        self.obl_min = self.comp_min / 10
        self.solid_sat = np.ones(self.nx) * 0.7

        # Several parameters here related to components used, OBL limits, and injection composition:
        self.cell_property = ['pressure', 'H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+',
                              'CaHCO3+', 'Solid']
        self.phases = ['liq', 'gas']
        self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.elements = ['Solid', 'Ca', 'C', 'O', 'H']
        Mw = [-1e+5] * 5 # dummy array
        self.num_vars = len(self.elements)
        self.n_points = 501
        self.min_p = 99
        self.max_p = 103
        self.min_z = self.obl_min
        self.max_z = 1 - self.obl_min

        # Rate annihilation matrix
        self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                      [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0],
                      [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]])

        # Several parameters related to kinetic reactions:
        stoich_matrix = np.array([-1, 1, 1, 3, 0])

        # Create instance of data-structure for simulation (and chemical) input parameters:
        input_data_struct = MyOwnDataStruct(len(self.elements), self.comp_min, self.temperature, stoich_matrix,
                                            self.pressure_init, self.c_r, self.kin_fact)

        # Create property containers:
        property_container = ModelProperties(phases_name=self.phases, components_name=self.elements,
                                                  Mw=Mw, min_z=self.comp_min, temperature=self.temperature)

        # Create instance of (own) physics class:
        self.physics = PhreeqcDissolution(self.timer, self.elements, self.n_points, self.min_p, self.max_p,
                                       self.min_z, input_data_struct, property_container)

        self.physics.add_property_region(property_container, 0)

        # set transmissibility exponent power
        self.params.trans_mult_exp = 4

        # Compute injection stream
        mole_water, mole_co2 = calculate_injection_stream(1.1, 0.1, self.temperature, self.pressure_init)
        mole_fraction_water, mole_fraction_co2 = get_mole_fractions(mole_water, mole_co2)

        # Define injection stream composition,
        # ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.inj_stream_components = np.array([mole_fraction_water, 0, 0, mole_fraction_co2, 0, 0, 0, 0, 0, 0, 0])
        self.inj_stream = convert_composition(self.inj_stream_components, self.E)
        self.inj_stream = correct_composition(self.inj_stream, self.comp_min)

        # prepare arrays for evaluation of properties
        nb = self.nx * self.ny * self.nz
        n_vars = self.physics.n_vars
        self.prop_states = value_vector([0.] * nb * (n_vars + 1))
        self.prop_states_np = np.asarray(self.prop_states)
        self.prop_values = value_vector([0.] * 2 * nb)
        self.prop_values_np = np.asarray(self.prop_values)
        self.prop_dvalues = value_vector([0.] * 2 * nb * n_vars)

    def set_reservoir(self, nx):
        self.nx = nx
        self.ny = 1
        self.nz = 1
        self.dx = 0.1 / nx      # m
        self.dy = 0.001000      # m
        self.dz = 0.058905      # m
        self.volume = self.dx * self.dy * self.dz
        self.depth = 1                      # m
        self.poro = 1                       # [-]
        self.const_perm = 1.25e4 * self.poro ** 4

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy,
                                         dz=self.dz, permx=self.const_perm, permy=self.const_perm,
                                         permz=self.const_perm, poro=self.poro, depth=self.depth)

    def set_initial_conditions(self):
        # ====================================== Initialize reservoir composition ======================================
        print('\nInitializing compositions...')

        # Component-defined composition of a non-solid phase (pure water here)
        self.initial_comp_components = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.solid_frac = np.zeros(len(self.solid_sat))
        self.initial_comp = np.zeros((self.nx * self.ny * self.nz + 4, self.num_vars - 1))

        # Interpolated values of non-solid volume (second value always 0 due to no (5,1) interpolator)
        values = value_vector([0] * 2)

        # Iterate over solid saturation and call interpolator
        for i in range(len(self.solid_sat)):
            # There are 5 values in the state
            composition_full = convert_composition(self.initial_comp_components, self.E)
            composition = correct_composition(composition_full, self.comp_min)
            init_state = value_vector(np.hstack((self.physics.input_data_struct.pressure_init, self.solid_sat[i], composition)))

            # Call interpolator
            self.physics.comp_itor.evaluate(init_state, values)

            # Assemble initial composition
            self.solid_frac[i] = values[0]
            initial_comp_with_solid = np.multiply(composition_full, 1 - self.solid_frac[i])
            initial_comp_with_solid[0] = self.solid_frac[i]
            self.initial_comp[i, :] = correct_composition(initial_comp_with_solid, self.comp_min)

        # Define initial composition for wells
        for i in range(self.nx * self.ny * self.nz, self.nx * self.ny * self.nz + 2):
            self.initial_comp[i, :] = np.array(self.inj_stream)

        for i in range(self.nx * self.ny * self.nz + 2, self.nx * self.ny * self.nz + 4):
            self.initial_comp[i, :] = np.array(self.initial_comp[0, :])

        print('\tNegative composition occurrence while initializing:', self.physics.property_operators[0].counter, '\n')

        initial_pressure = self.pressure_init
        initial_composition = self.initial_comp

        nb = self.reservoir.mesh.n_blocks
        nc = self.physics.nc

        # set initial pressure
        pressure = np.array(self.reservoir.mesh.pressure, copy=False)
        pressure.fill(initial_pressure)

        # set initial composition
        self.reservoir.mesh.composition.resize(nb * (nc - 1))
        composition = np.array(self.reservoir.mesh.composition, copy=False)
        for c in range(nc - 1):
            composition[c::nc-1] = initial_composition[:, c]

    def set_wells(self):
        d_w = 1.5
        r_w = d_w / 2
        well_index = 5

        self.reservoir.add_well("I1", wellbore_diameter=d_w)
        for idx in range(self.ny):
            self.reservoir.add_perforation(well_name='I1', cell_index=(1, 1, 1), multi_segment=False,
                                           verbose=True, well_radius=r_w, well_index=well_index,
                                           well_indexD=well_index)

        self.reservoir.add_well("P1", wellbore_diameter=d_w)
        for idx in range(self.ny):
            self.reservoir.add_perforation(well_name='P1', cell_index=(self.nx, 1, 1), multi_segment=False,
                                           verbose=True, well_radius=r_w, well_index=well_index,
                                           well_indexD=well_index)

    def set_boundary_conditions(self):
        # New boundary condition by adding wells:
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
                w.control = self.physics.new_rate_oil_inj(self.inj_rate, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(100)

    def set_op_list(self):
        """
        Function to define list of operator interpolators for accumulation-flux regions and wells.

        Operator list is in order [acc_flux_itor[0], ..., acc_flux_itor[n-1], acc_flux_w_itor]
        """
        self.op_list = [self.physics.acc_flux_itor] + [self.physics.acc_flux_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks:] = len(self.op_list) - 1

    def evaluate_porosity(self):
        # Initial porosity
        # poro_init = 1 - self.solid_sat
        nb = self.nx * self.ny * self.nz
        n_vars = self.physics.n_vars

        X = np.asarray(self.physics.engine.X)
        self.prop_states_np[0::n_vars + 1] = X[0:nb * n_vars:n_vars]
        for i in range(n_vars):
            self.prop_states_np[i+2::n_vars + 1] = X[i+1:nb * n_vars:n_vars]

        region = 0
        self.physics.comp_itor.evaluate_with_derivatives(self.prop_states, self.physics.engine.region_cell_idx[region],
                                                         self.prop_values, self.prop_dvalues)
        poro = 1. - self.prop_values_np[1::2]

        print('\tNegative composition while evaluating results:', self.physics.property_operators[0].counter, '\n')
        return poro

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, nc_sol=0, np_sol=0, min_z=1e-11, rock_comp=1e-6,
                 rate_ann_mat=None, temperature=None):
        super().__init__(phases_name, components_name, Mw, nc_sol, np_sol, min_z, rock_comp, rate_ann_mat, temperature)

        # Define custom evaluators
        self.flash_ev = self.CustomFlash(self.temperature, self.min_z)
        self.init_flash_ev = self.InitFlash(self.temperature, self.min_z)
        self.kinetic_rate_ev = self.CustomKineticRate(self.temperature, self.min_z)
        self.rel_perm_ev = {ph: self.CustomRelPerm(2) for ph in phases_name[:2]}  # Relative perm for first two phases

    class CustomFlash:
        def __init__(self, temperature, comp_min):
            self.temperature = temperature - 273.15
            self.comp_min = comp_min
            self.phreeqc = IPhreeqc()
            self.load_database("phreeqc.dat")
            # self.phreeqc.phreeqc.OutputFileOn = True
            # self.phreeqc.phreeqc.SelectedOutputFileOn = True

            self.phreeqc_template = """
            USER_PUNCH
            -headings    H(mol)      O(mol)      C(mol)      Ca(mol)      Vol_aq   SI            SR            ACT("H+") ACT("CO2") ACT("H2O")
            10 PUNCH    TOTMOLE("H") TOTMOLE("O") TOTMOLE("C") TOTMOLE("Ca") SOLN_VOL SI("Calcite") SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O")

            SELECTED_OUTPUT
            -selected_out    true
            -user_punch      true
            -reset           false
            -high_precision  true
            -gases           CO2(g) H2O(g)

            SOLUTION 1
            temp      {temperature:.2f}
            pressure  {pressure:.4f}
            pH        7 charge
            -water    {water_mass:.10f} # kg
            REACTION 1
            H         {hydrogen:.10f}
            O         {oxygen:.10f}
            C         {carbon:.10f}
            Ca        {calcium:.10f}
            1
            KNOBS
            -convergence_tolerance  1e-10
            END
            """

        def load_database(self, db_path):
            try:
                self.phreeqc.load_database(db_path)
            except Exception as e:
                warnings.warn(f"Failed to load '{db_path}': {e}.", Warning)

        @staticmethod
        def get_moles(composition):
            # Assume 1000 mol of solution
            total_mole = 1000
            hydrogen_mole = total_mole * composition[-1]
            oxygen_mole = total_mole * composition[-2]
            carbon_mole = total_mole * composition[-3]
            calcium_mole = total_mole * composition[-4]
            return hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

        def get_composition(self, state):
            comp = state[2:]
            # comp[comp <= self.comp_min / 10] = 0
            comp = np.divide(comp, np.sum(comp))
            return comp

        @staticmethod
        def generate_input(*args):
            hydrogen, oxygen, carbon, calcium = args[0], args[1], args[2], args[3]
            if hydrogen / 2 <= oxygen:
                water_mass = hydrogen / 2 * 0.018016
                hydrogen_mole = 0
                oxygen_mole = oxygen - hydrogen / 2
            else:
                water_mass = oxygen * 0.018016
                hydrogen_mole = hydrogen - 2 * oxygen
                oxygen_mole = 0
            carbon_mole = carbon
            calcium_mole = calcium
            return water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

        def interpret_results(self):
            results_array = np.array(self.phreeqc.get_selected_output_array()[2])

            # Interpret aqueous phase
            hydrogen_mole_aq = results_array[5]
            oxygen_mole_aq = results_array[6]
            carbon_mole_aq = results_array[7]
            calcium_mole_aq = results_array[8]

            volume_aq = results_array[9] / 1000  # m3
            total_mole_aq = (hydrogen_mole_aq + oxygen_mole_aq + carbon_mole_aq + calcium_mole_aq)  # mol
            rho_aq = total_mole_aq / volume_aq / 1000  # kmol/m3

            x = np.array([0,
                          calcium_mole_aq / total_mole_aq,
                          carbon_mole_aq / total_mole_aq,
                          oxygen_mole_aq / total_mole_aq,
                          hydrogen_mole_aq / total_mole_aq])

            # Interpret gaseous phase
            y = np.zeros(len(x))
            rho_g = 0
            total_mole_gas = 0

            rho_phases = {'aq': rho_aq, 'gas': rho_g}
            vap = total_mole_gas / (total_mole_aq + total_mole_gas)

            # Interpret kinetic parameters
            kin_state = {'SI': results_array[10],
                         'SR': results_array[11],
                         'Act(H+)': results_array[12],
                         'Act(CO2)': results_array[13],
                         'Act(H2O)': results_array[14]}
            return vap, x, y, rho_phases, kin_state

        def evaluate(self, state):
            pressure_atm = bar2atm(state[0])
            comp = self.get_composition(state)
            hydrogen, oxygen, carbon, calcium = self.get_moles(comp)
            water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole = self.generate_input(hydrogen, oxygen,
                                                                                                    carbon, calcium)
            # Generate and execute PHREEQC input
            input_string = self.phreeqc_template.format(
                temperature=self.temperature,
                pressure=pressure_atm,
                water_mass=water_mass,
                hydrogen=hydrogen_mole,
                oxygen=oxygen_mole,
                carbon=carbon_mole,
                calcium=calcium_mole
            )

            try:
                self.phreeqc.run_string(input_string)
            except Exception as e:
                warnings.warn(f"Failed to run PHREEQC: {e}", Warning)

            vap, x, y, rho_phases, kin_state = self.interpret_results()
            vap = vap * (1 - state[1])
            return vap, x, y, rho_phases, kin_state

    class InitFlash(CustomFlash):
        def __init__(self, temperature, comp_min):
            super().__init__(temperature, comp_min)

        @staticmethod
        def get_moles(comp_data):
            # Assume 1000 mol of solution
            total_mole = comp_data[-1]
            hydrogen_mole = total_mole * comp_data[-2]
            oxygen_mole = total_mole * comp_data[-3]
            carbon_mole = total_mole * comp_data[-4]
            calcium_mole = total_mole * comp_data[-5]
            return hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

        def evaluate(self, state):
            # Convert pressure to atm
            pressure_atm = bar2atm(state[0])

            # Normalize the non-solid composition values
            non_solid_composition = np.divide(state[3:], np.sum(state[3:]))
            comp_data = np.hstack((non_solid_composition, state[1]))

            # Calculate moles for each element based on comp_data
            hydrogen, oxygen, carbon, calcium = self.get_moles(comp_data)
            water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole = self.generate_input(
                hydrogen, oxygen, carbon, calcium
            )

            # Create and run the input string using phreeqpy
            input_string = f"""
            USER_PUNCH
            -headings	Vol_aq(L)
            10 PUNCH	SOLN_VOL

            SELECTED_OUTPUT
                -selected_out    true
                -user_punch      true
                -reset           false
                -high_precision  true
            SOLUTION 1
                temp      {self.temperature:.2f}
                pressure  {pressure_atm:.4f}
                pH        7 charge
                -water    {water_mass:.3f} # kg
            REACTION 1
                H         {hydrogen_mole:.8f}
                O         {oxygen_mole:.8f}
                C         {carbon_mole:.8f}
                Ca        {calcium_mole:.8f}
                1
            END
            """

            try:
                # Run PHREEQC with the formatted input string
                self.phreeqc.run_string(input_string)
            except Exception as e:
                warnings.warn(f"Failed to run PHREEQC: {e}", Warning)

            # Retrieve the non-solid volume output from PHREEQC
            non_solid_volume = np.array(self.phreeqc.get_selected_output_array()[2]) / 1000  # Convert to m3
            return non_solid_volume

    class CustomKineticRate:
        def __init__(self, temperature, comp_min):
            self.temperature = temperature
            self.comp_min = comp_min

        def evaluate(self, kin_state, solid_saturation, rho_s, min_z, kin_fact):
            # Define constants
            specific_sa = 0.925         # [m2/mol], default = 0.925
            k25a = 0.501187234          # [mol * m-2 * s-1]
            k25n = 1.54882e-06          # [mol * m-2 * s-1]
            Eaa = 14400                 # [J * mol-1]
            Ean = 23500                 # [J * mol-1]
            na = 1                      # reaction order with respect to H+
            R = 8.314472                # gas constant [J/mol/Kelvin]
            p = 1
            q = 1
            sat_ratio_threshold = 100

            # Define rate parameters
            sat_ratio = min(kin_state['SR'], sat_ratio_threshold)
            hydrogen_act = kin_state['Act(H+)']
            KTa = k25a * np.exp((-Eaa / R) * (1 / self.temperature - 1 / 298.15)) * hydrogen_act ** na
            KTn = k25n * np.exp((-Ean / R) * (1 / self.temperature - 1 / 298.15))

            # [mol/s]
            kinetic_rate = -specific_sa * solid_saturation * (rho_s * 1000) * (KTa + KTn) * (1 - sat_ratio ** p) ** q

            # [kmol/d]
            kinetic_rate *= 60 * 60 * 24 / 1000
            return kinetic_rate

    class CustomRelPerm:
        def __init__(self, exp, sr=0):
            self.exp = exp
            self.sr = sr

        def evaluate(self, sat):
            return (sat - self.sr) ** self.exp


